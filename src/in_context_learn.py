import logging
from gemini_cot import generate_response 
from typing import Union 


class InContextLearner:
    def __init__(self, agent):
        """Initializes the learner with a reference to the agent."""
        self.agent = agent 
        self.prompts = agent.prompts
        logging.debug("InContextLearner initialized.")

    # Removed hf_pipeline argument
    def chain_of_thought(self) -> Union[str, None]:
        """ 
        Generates a land use decision using a Chain-of-Thought prompt with Gemini.
        Returns the raw text response from Gemini, or None on failure.
        """
        logging.debug("Starting Chain-of-Thought decision process.")
        
        # --- Construct the Prompt for Gemini --- 
        # System instruction (describes the agent's role and overall goal)
        # Ensure self.prompts.system_prompt contains the high-level instructions
        system_instruction = getattr(self.prompts, 'system_prompt', "You are a farmer making a land use decision.")

        # Detailed User Prompt combining context and task
        user_prompt_parts = []

        # 1. Yearly context (market, news, policy)
        yearly_prompt = getattr(self.prompts, 'yearly_prompt', "No specific yearly information provided.")
        user_prompt_parts.append("== Current Year Information ==")
        user_prompt_parts.append(yearly_prompt)

        # 2. Previous context (past decisions, outcomes)
        prev_context_prompt = getattr(self.prompts, 'prev_context_prompt', "No previous context provided.")
        user_prompt_parts.append("\n== Previous Context ==")
        user_prompt_parts.append(prev_context_prompt)
        
        # 3. Computed details for possible decisions
        compute_prompt = getattr(self.prompts, 'compute_prompt', "No computed decision details provided.")
        user_prompt_parts.append("\n== Potential Decision Outcomes (Computed) ==")
        user_prompt_parts.append(compute_prompt)

        # 4. The Chain-of-Thought Task Instruction
        cot_instruction = (
            "\n== Task ==\n" 
            "Based *only* on the information provided above (Current Year Info, Previous Context, Potential Outcomes), "
            "think step-by-step to decide on the single best land use for this year. "
            "Consider your goals and traits (implicitly included in the system prompt and context). "
            "Explain your reasoning clearly, breaking down the problem. "
            "Finally, state your final choice explicitly.\n\n" 
            "Reasoning Steps:\n" 
            "1. Analyze the computed outcomes for each potential land use.\n" 
            "2. Evaluate these outcomes considering the current year information (market, policy, news).\n" 
            "3. Reflect on how past decisions and outcomes influence this year's choice.\n" 
            "4. Synthesize these factors to arrive at a reasoned decision.\n\n" 
            "Output Format:\n" 
            "Start with your step-by-step reasoning process.\n" 
            "Conclude with the final decision clearly marked on a new line like this:\n" 
            "Final decision: [Chosen Land Use Name] - Brief justification (1-2 sentences)."
        )
        user_prompt_parts.append(cot_instruction)

        # Combine parts into the final user prompt
        full_user_prompt = "\n".join(user_prompt_parts)
        
        logging.debug(f"System Instruction: {system_instruction}")
        logging.debug(f"User Prompt: {full_user_prompt[:500]}...") # Log beginning of prompt

        # --- Call Gemini API --- 
        # Uses the generate_response function from gemini_cot
        response_text = generate_response(
            system_instruction=system_instruction,
            prompt=full_user_prompt
        )

        if response_text:
            logging.info("Received response from Gemini.")
        else:
            logging.error("Failed to get response from Gemini in chain_of_thought.")

        return response_text
