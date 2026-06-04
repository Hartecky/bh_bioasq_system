from tqdm import tqdm
from src.agents.pipeline import QAPipeline
from src.data.dataset import Question


class Evaluator:
    def __init__(self, pipeline: QAPipeline):
        self.pipeline = pipeline

    def evaluate(self, questions: list[Question]) -> dict:
        """Uruchamia pipeline na wszystkich pytaniach i zwraca metryki per typ."""

        # Słownik do zbierania wyników per typ
        results = {
            "yesno":   {"correct": 0, "total": 0},
            "factoid": {"correct": 0, "total": 0},
            "list":    {"f1": [], "total": 0},
            "summary": {"rouge": [], "total": 0},
        }

        for q in tqdm(questions, desc="Evaluating"):
            # Pomiń pytania bez golden standard
            if not q.exact_answer and not q.ideal_answer:
                continue

            predicted = self.pipeline.answer(q)

            if q.type == "yesno":
                correct = self._eval_yesno(predicted, q.exact_answer)
                results["yesno"]["correct"] += int(correct)
                results["yesno"]["total"] += 1

            elif q.type == "factoid":
                correct = self._eval_factoid(predicted, q.exact_answer)
                results["factoid"]["correct"] += int(correct)
                results["factoid"]["total"] += 1

            elif q.type == "list":
                f1 = self._eval_list(predicted, q.exact_answer)
                results["list"]["f1"].append(f1)
                results["list"]["total"] += 1

            elif q.type == "summary":
                rouge = self._eval_summary(predicted, q.ideal_answer)
                results["summary"]["rouge"].append(rouge)
                results["summary"]["total"] += 1

        return self._compute_final_metrics(results)

    def _eval_yesno(self, predicted: str, expected) -> bool:
        """
        Exact match dla yes/no.
        Normalizujemy oba stringi — model może zwrócić "Yes" lub " yes ".
        """
        pred = predicted.lower().strip()
        gold = str(expected).lower().strip() if expected else ""

        # Bierzemy tylko pierwsze słowo — model czasem pisze "yes, because..."
        pred_word = pred.split()[0] if pred else ""
        return pred_word == gold

    def _eval_factoid(self, predicted: str, expected) -> bool:
        """
        Lenient match — czy odpowiedź modelu zawiera którykolwiek z expected.
        Lenient (łagodny) zamiast strict (dokładny) bo model może parafrazować.
        """
        pred = predicted.lower().strip()

        if isinstance(expected, list):
            golds = [str(g).lower().strip() for g in expected]
        else:
            golds = [str(expected).lower().strip()]

        # Sprawdź czy którykolwiek gold jest zawarty w odpowiedzi
        return any(gold in pred for gold in golds)

    def _eval_list(self, predicted: str, expected) -> float:
        """
        F1 score dla list.
        F1 = harmonic mean precision i recall.
        Precision = ile z predicted jest w golden / ile predicted
        Recall    = ile z golden znalazł model / ile golden
        """
        # Parsuj odpowiedź modelu — dzielimy po przecinkach i newlines
        pred_items = set(
            item.lower().strip()
            for item in predicted.replace("\n", ",").split(",")
            if item.strip()
        )

        # Parsuj golden standard
        if isinstance(expected, list):
            gold_items = set(str(g).lower().strip() for g in expected)
        else:
            gold_items = {str(expected).lower().strip()}

        if not pred_items or not gold_items:
            return 0.0

        tp = len(pred_items & gold_items)  # & = przecięcie zbiorów
        precision = tp / len(pred_items)
        recall = tp / len(gold_items)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _eval_summary(self, predicted: str, expected) -> float:
        """
        ROUGE-2 dla summary.
        ROUGE mierzy ile n-gramów z golden pojawia się w odpowiedzi modelu.
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

            gold = expected[0] if isinstance(expected, list) else str(expected)
            score = scorer.score(gold, predicted)
            return score["rouge2"].fmeasure

        except ImportError:
            print("Install rouge-score: uv add rouge-score")
            return 0.0

    def _compute_final_metrics(self, results: dict) -> dict:
        """Agreguje wyniki do finalnych metryk."""
        metrics = {}

        if results["yesno"]["total"] > 0:
            n = results["yesno"]["total"]
            metrics["yesno_accuracy"] = results["yesno"]["correct"] / n
            metrics["yesno_total"] = n

        if results["factoid"]["total"] > 0:
            n = results["factoid"]["total"]
            metrics["factoid_accuracy"] = results["factoid"]["correct"] / n
            metrics["factoid_total"] = n

        if results["list"]["total"] > 0:
            scores = results["list"]["f1"]
            metrics["list_f1"] = sum(scores) / len(scores)
            metrics["list_total"] = results["list"]["total"]

        if results["summary"]["total"] > 0:
            scores = results["summary"]["rouge"]
            metrics["summary_rouge2"] = sum(scores) / len(scores)
            metrics["summary_total"] = results["summary"]["total"]

        return metrics