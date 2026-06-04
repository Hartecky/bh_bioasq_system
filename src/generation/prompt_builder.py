class PromptBuilder():
    def __init__(self):
        pass

    def build(self, question, context, question_type):

        builders = {
            "yesno":   self._yesno_prompt,
            "factoid": self._factoid_prompt,
            "list":    self._list_prompt,
            "summary": self._summary_prompt,
        }

        return builders[question_type](question, context)
