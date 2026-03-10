














































































































































































































































































































































































































        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from ..core.prompts import build_critic_prompt, CRITIC_SYSTEM_PROMPT

        # Step 1: Get free-form evaluation from LLM
        prompt = build_critic_prompt(llmstxt, url, source_content)
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            evaluation_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"[SimpleDraftCritic] Evaluation text ({len(evaluation_text)} chars)")
        except Exception as e:
            raise ValueError(f"LLM call failed during evaluation: {e}")

        # Step 2: Extract structured result using JSON parsing with explicit prompting
        extraction_prompt = f"""Based on the evaluation, extract the structured result as JSON.

        Evaluation text:
        ---
        {evaluation_text}
        ---

        The output must match this JSON schema:
        {{
            "passed": boolean,
            "score": float (0.0-1.0),
            "issues": ["string"],
            "suggestions": ["string"]
        }}

        IMPORTANT:
        - Return ONLY valid JSON (no markdown, no code blocks)
        - Be strict: passed=True only if ALL rules are satisfied
        - Include all issues found, even minor ones
        """

        extraction_messages = [
            HumanMessage(content=extraction_prompt),
        ]

        try:
            extraction_response = await self.llm.ainvoke(extraction_messages)
            extraction_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
            logger.debug(f"[SimpleDraftCritic] Extraction text ({len(extraction_text)} chars)")
        except Exception as e:
            raise ValueError(f"LLM call failed during extraction: {e}")

        # Parse JSON response
        import json
        import re

        text = extraction_text.strip()
        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
            result = CriticResult(**data)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON. "
                f"Expected format: {{\"passed\": bool, \"score\": float, \"issues\": [...], \"suggestions\": [...]}}. "
                f"Got: {text[:200]}..."
            )

        # Apply threshold override
        if result.score >= self.config.pass_threshold and not result.passed:
            logger.info(
                f"[SimpleDraftCritic] Overriding passed=True "
                f"(score {result.score:.2f} >= threshold {self.config.pass_threshold})"
































































































