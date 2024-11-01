from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


class InContextLearner:
    def __init__(self, agent):
        self.prompts = agent.prompts
        self.compute = agent.compute

    def chain_of_thought(self, hf_pipeline):
        cot_prompt = ("Task1: You will be provided with results from an optimization run "
                      "computing the expected profit or loss made, costs, and labour and technology required "
                      "for each of the land use decisions this year. "
                      "Based on these results you need to brainstorm a reasoning chain "
                      "for a final land use decision for the year. Think step by step. "
                      "Break the problem into sub-problems and solve those sub-problems. "
                      "Keep all previous instructions in mind. "
                      "Provide the final decision in the following format - "
                      "Final decision: <Your land use decision> - 3 sentence Justification")

        compute_cot_template = ("<|system|> {system_prompt} {yearly_prompt} {prev_context_prompt} </s> "
                                "<|user|> {cot_prompt} {compute_prompt} </s> "
                                "<|assistant|>")

        compute_cot_prompt_template = PromptTemplate(
            input_variables=["system_prompt", "yearly_prompt", "prev_context_prompt", "cot_prompt", "compute_prompt"],
            template=compute_cot_template
        )

        compute_reasoning_chain = LLMChain(
            llm=hf_pipeline,
            prompt=compute_cot_prompt_template,
            output_key="lu_decision"
        )

        """
        decision_template = ("<|user|> {reasoning_from_compute} Task2: For the proposed reasoning chain, "
                             "evaluate its potential rigorously given all previous instructions. "
                             "Provide the final decision in the following format - "
                             "Final decision: <Your land use decision> - 3 sentence Justification."
                             "and keep going. </s> <|assistant|>")

        decision_prompt_template = PromptTemplate(
            input_variables=["reasoning_from_compute"],
            template=decision_template
        )

        decision_chain = LLMChain(
            llm=hf_pipeline,
            prompt=decision_prompt_template,
            output_key="lu_decision"
        )
        """

        """
        decision_template = ("<|user|> {review} "
                             "Task3: Based on the computations, reasoning and review, identify the "
                             "final decision that is promising. Keep all previous instructions in mind. "
                             "Provide the final decision in the following format - "
                             "Final decision: <Your land use decision> - 3 sentence Justification. "
                             "</s> <|assistant|> ")

        decision_prompt_template = PromptTemplate(
            input_variables=["review"],
            template=decision_template
        )

        decision_chain = LLMChain(
            llm=hf_pipeline,
            prompt=decision_prompt_template,
            output_key="lu_decision"
        )
        """

        overall_chain = SequentialChain(
            chains=[compute_reasoning_chain],
            input_variables=["system_prompt", "yearly_prompt", "prev_context_prompt", "cot_prompt", "compute_prompt"],
            output_variables=["lu_decision"],
            verbose=True)

        response = overall_chain(
            {"system_prompt": self.prompts.system_prompt, "yearly_prompt": self.prompts.yearly_prompt,
             "prev_context_prompt": self.prompts.prev_context_prompt, "cot_prompt": cot_prompt,
             "compute_prompt": self.prompts.compute_prompt})
        return response

    def tree_of_thought(self):
        init_prompt = ("Task1: You need to brainstorm three alternate reasoning chains. "
                       "They can be for the same final land use decision or different ones. "
                       "Think step by step. Break the problem into sub-problems and solve those sub-problems.")

        chain1_template = ("<|system|> {system_prompt} </s> <|user|> {init_prompt} </s> "
                           "<|assistant|>")

        chain1_prompt = PromptTemplate(
            input_variables=["system_prompt", "init_prompt"],
            template=chain1_template
        )

        chain1 = LLMChain(
            llm=self.agent.hf,
            prompt=chain1_prompt,
            output_key="reasoning_chains"
        )

        chain2_template = ("<|user|> {reasoning_chains} Task3: compute using functions "
                           "</s> <|assistant|>")

        chain2_prompt = PromptTemplate(
            input_variables=["reasoning_chains"],
            template=chain2_template
        )

        chain2 = LLMChain(
            llm=self.agent.hf,
            prompt=chain2_prompt,
            output_key="computed_outputs"
        )

        chain3_template = ("<|user|> {computed_outputs} Task2: For each of the three proposed reasoning chains, "
                           "evaluate their potential given the computed details. "
                           "Consider their pros and cons, implementation difficulty, potential challenges, and the "
                           "expected outcomes. Generate potential scenarios, strategies for implementation, "
                           "any necessary partnerships or resources, and how potential obstacles might be overcome. "
                           "Also consider how the computed details align with who you are, "
                           "what you value, and all other information given to you. Keep your system prompt in mind. "
                           "Assign a probability of success. </s> <|assistant|>")

        chain3_prompt = PromptTemplate(
            input_variables=["computed_outputs"],
            template=chain3_template
        )

        chain3 = LLMChain(
            llm=self.agent.hf,
            prompt=chain3_prompt,
            output_key="review"
        )

        chain4_template = ("<|user|> {review} "
                           "Task4: Based on the computations and review, rank the solutions in order "
                           "of promise. Keep your system prompt in mind. "
                           "Provide the final winning decision in the following format-"
                           "Final decision: Land use decision - 3 sentence Justification."
                           "</s> <|assistant|> ")

        chain4_prompt = PromptTemplate(
            input_variables=["review"],
            template=chain4_template
        )

        chain4 = LLMChain(
            llm=self.agent.hf,
            prompt=chain4_prompt,
            output_key="final_decision"
        )

        overall_chain = SequentialChain(
            chains=[chain1, chain2, chain3, chain4],
            input_variables=["system_prompt", "init_prompt"],
            output_variables=["final_decision"],
            verbose=True)

        response = overall_chain({"system_prompt": self.agent.system_prompt, "init_prompt": init_prompt})
        return response

    def teach_using_examples(self):
        pass
