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
    
    def _yesno_prompt(self, question, context):
        prompt = f"""
        Based on the context below answer the question with ONLY 'yes' or 'no'
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        return prompt
    
    def _factoid_prompt(self, question, context):
        prompt = f"""
        Based on the context below answer the question providing a fact in a single short phrase or name
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        return prompt
    
    def _list_prompt(self, question, context):
        prompt = f"""
        Based on the context below answer the question providing as a comma separated list
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        return prompt
    
    def _summary_prompt(self, question, context):
        prompt = f"""
        Based on the context below answer the question providing a comprehensive summary
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        return prompt

