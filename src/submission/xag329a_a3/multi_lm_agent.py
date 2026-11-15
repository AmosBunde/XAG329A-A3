from typing import Dict, Any, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor
from xag329a_a3.api_manager import APIManager
from xag329a_a3.utils import generate_llm

DEFAULT_FUSION_SAMPLE_MODELS = (
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/openai/gpt-oss-20b",
)


def _summarize_results(results: Any) -> Any:
    """Drops `tavily_search`'s full page text, keeping its snippet.

    Search results carry both `snippet` and the whole `webpage_content`, and the
    refinement history is re-serialized on every iteration, so passing pages
    through crowds out the reasoning the model needs to do.
    """
    if isinstance(results, list):
        return [
            {key: value for key, value in item.items() if key != "webpage_content"}
            if isinstance(item, dict) else item
            for item in results
        ]
    return results


def _step_result(sub_query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Records one sub-query and the outcome `route_query` reported for it."""
    step = {
        "sub_query": sub_query,
        "api": result.get("api_used"),
        "params": result.get("params"),
        "results": result.get("results"),
    }
    if "error" in result:
        step["error"] = result["error"]
    return step


class MultiLMAgent:
    """A class to manage multiple language models for generation, iterative refinement, and fusion"""
    
    def __init__(
        self,
        api_manager: APIManager,
        decomposition_model: str = "openrouter/meta-llama/llama-3.3-70b-instruct",
        iterative_refinement_model: str = "openrouter/meta-llama/llama-3.3-70b-instruct",
        fusion_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
        fusion_sample_models: Sequence[str] = DEFAULT_FUSION_SAMPLE_MODELS,
        generation_temp: float = 0.7,
    ):
        """Initializes the MultiLMAgent.

        Args:
            api_manager (APIManager): An instance of APIManager for API interactions.
            decomposition_model (str): Model used for query decomposition.
            iterative_refinement_model (str): Model used for iterative refinement.
            fusion_model (str): Model used for fusing responses.
            fusion_sample_models (Sequence[str]): Models sampled for fusion.
            generation_temp (float): Temperature for language model generation.
        """
        ### TODO: YOUR CODE STARTS HERE
        self.api_manager = api_manager
        self.decomposition_model = decomposition_model
        self.iterative_refinement_model = iterative_refinement_model
        self.fusion_model = fusion_model
        self.fusion_sample_models = tuple(fusion_sample_models)
        self.generation_temp = generation_temp
        ### TODO: YOUR CODE ENDS HERE

    def generate(
        self, 
        query: str, 
        model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
        temperature: None | float = None,
    ) -> Optional[str]:
        """Generates a response to the query using the specified model, or None if the call fails."""
        if temperature is None:
            temperature = self.generation_temp

        # Hint: use `generate_llm` function from `utils.py`.
        # Hint: https://docs.litellm.ai/docs/providers/openrouter or https://openrouter.ai/docs/quickstart

        ### TODO: YOUR CODE STARTS HERE
        messages = [{"role": "user", "content": query}]
        response = generate_llm(model, messages, temperature)
        ### TODO: YOUR CODE ENDS HERE

        # `generate_llm` returns None once its retries are exhausted (e.g. a rate
        # limit), so every caller below has to tolerate a missing response rather
        # than let one throttled request abort a whole evaluation run.
        if not response:
            print(f"Generation failed for {model}.")
            return None
        return response
    
    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """Generates a response by querying the API manager for the necessary data and then using an LM to generate a final response based on the API output."""

        # HINT: This is the whole pipeline in two moves: one tool call, then one LM call.
        # The API manager's `route_query` already picks a tool, fills in its arguments,
        # and runs it, so there is no tool choosing to do here -- what comes back is the
        # raw tool output along with which tool produced it.
        # Then have an LM turn that raw output into an answer, via `self.generate`.
        # Give it both the original question and the tool's output, since neither alone is
        # enough, and ask it to state the final answer plainly at the end: `evaluate_qa`
        # grades the text this returns.
        # prompt_help.md has a prompt suggestion for the routing call.
        ### TODO: YOUR CODE STARTS HERE
        api_result = self.api_manager.route_query(query) or {}
        api_used = api_result.get("api_used")
        api_output = api_result.get("results")
        api_error = api_result.get("error")

        prompt = "Here is the query: " + query + "\n"
        source = f" from the '{api_used}' tool" if api_used else ""
        prompt += f"Here is the response from the API{source}: " + str(api_output) + "\n"
        if api_error:
            prompt += (
                "The API call reported an error: " + str(api_error) + "\n"
                "Answer the query as well as you can without the API data, and say "
                "clearly if the answer cannot be determined.\n"
            )
        prompt += (
            "Generate a concise and accurate response to the query based on the API "
            "response. Clearly state your final answer at the end of the response."
        )

        response = self.generate(prompt, model=model)
        if not response:
            if api_error:
                return f"Error: the API call failed ({api_error}) and no answer could be generated."
            return "Error: no answer could be generated for the query."
        return response
        ### TODO: YOUR CODE ENDS HERE

    def _get_query_decomposition_prompt(self, query: str) -> str:
        """Helper function to generate a prompt for the LLM to decompose the query into sub-queries."""

        # Hint: A helpful way structure the output is to enclose each sub-query in <sub-query>...</sub-query> tags. You should provide the model with instructions on how to decompose the query into sub-queries.
        # prompt_help.md has a prompt suggestion
        ### TODO: YOUR CODE STARTS HERE
        api_functions = getattr(self.api_manager, "api_functions", {}) or {}
        api_names = ", ".join(api_functions.keys()) or "none"

        prompt = f"""
Your task is to break down the following user query into the minimal set of sub-queries needed for the API manager to answer it effectively.

User query:
{query}

Available API functions:
{api_names}

Guidelines:
- Each sub-query should be enclosed in <sub-query>...</sub-query> tags.
- Sub-queries should be distinct and complementary, covering different aspects of the user's request.
- Only include as many sub-queries as are truly necessary to fully answer the user query.
- Sub-queries should NOT be dependent on each other; they will be executed independently, in parallel.
- The sub-queries should be able to be answered without relying on the results of other sub-queries.
- Each sub-query is answered by one API call with one set of parameters, so each sub-query must ask for exactly one fact: one city, one date, one ticker, or one calculation. A sub-query about two cities or three dates will only return the first one.
- Copy dates, tickers, and place names exactly as they appear in the user query. Do not substitute a nearby date or a different spelling.
- Output only the tagged sub-queries, nothing else.
"""
        return prompt
        ### TODO: YOUR CODE ENDS HERE

    def _get_sub_queries(self, query: str, max_sub_queries: int = 4) -> List[str]:
        """Helper function to break down a complex user query into smaller subqueries."""

        decomposition_response = None
        # HINT: Only two things happen here: build the prompt with
        # `_get_query_decomposition_prompt` above, then send it with `self.generate`.
        # Pass the decomposition model chosen in `__init__` rather than letting `generate`
        # Leave the reply as the raw text it comes back as: the code below is what reads
        # the sub-queries out of the tags and caps how many are kept.
        ### TODO: YOUR CODE STARTS HERE
        prompt = self._get_query_decomposition_prompt(query)
        decomposition_response = self.generate(prompt, model=self.decomposition_model)
        ### TODO: YOUR CODE ENDS HERE

        if decomposition_response is None:
            return []
        sub_queries = [q.strip() for q in decomposition_response.split("<sub-query>") if "</sub-query>" in q]
        sub_queries = [q.split("</sub-query>")[0].strip() for q in sub_queries]
        sub_queries = [q for q in sub_queries if q][:max_sub_queries]
        return sub_queries

    
    def decompose_query(self, query: str, max_sub_queries: int = 4) -> List[Dict[str, Any]]:
        """Decomposes a query into independent, more manageable sub-queries that are executed in parallel via the API manager.

        Args:
            query (str): The user's original query.
            max_sub_queries (int): The maximum number of sub-queries to generate.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each contains the
            sub-query, the API used to answer the sub-query, parameters, results, and any errors.
        """

        sub_queries = self._get_sub_queries(query, max_sub_queries=max_sub_queries)
        # No sub-queries means the decomposition call failed or returned no
        # <sub-query> tags; ThreadPoolExecutor rejects max_workers=0.
        if not sub_queries:
            return []
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
            # The sub-queries are independent, so route them all at once.
            sub_query_results = list(executor.map(
                lambda q: self.api_manager.route_query(q),
                sub_queries
            ))
        # Each step carries the API and parameters alongside the results, and an error
        # key when routing, parameter parsing, or the API call itself failed.
        return [
            _step_result(sub_query, sub_query_result)
            for sub_query, sub_query_result in zip(sub_queries, sub_query_results)
        ]

    def _get_synthesis_prompt(self, query: str, decomposed_queries: Sequence[Dict[str, Any]] = ()) -> str:
        """
        Constructs a prompt for an LLM to synthesize a final response to the user's query, using the API results of the decomposed sub-queries.
        """

        decomposed_queries_results_str = ""

        # HINT: `decomposed_queries` is what `decompose_query` handed back: one dictionary
        # per sub-query, holding the sub-query, the API used, its parameters, and the results.
        # The model reading this prompt never saw those API calls, so the prompt has to
        # carry both the original query and the gathered results.
        # Label each piece of evidence so the model can tell the results apart, and leave
        # out any step that came back with an `error` key, which has nothing to contribute.
        # Decomposition can also come back empty, so this should still produce a sensible
        # prompt when there are no results to attach.
        ### TODO: YOUR CODE STARTS HERE
        # Steps that carry an "error" key contributed nothing, so they are dropped
        # entirely: their sub-query, API name, parameters and results all stay out.
        successful_steps = [
            step for step in decomposed_queries
            if isinstance(step, dict) and "error" not in step
        ]
        for i, step in enumerate(successful_steps):
            decomposed_queries_results_str += f"Evidence {i + 1}:\n"
            decomposed_queries_results_str += f"Sub-query: {step.get('sub_query')}\n"
            decomposed_queries_results_str += f"Source: {step.get('api')} with parameters {step.get('params')}\n"
            decomposed_queries_results_str += f"Results: {_summarize_results(step.get('results'))}\n\n"

        prompt = f"Your task is to answer the following query: {query}\n"
        if successful_steps:
            prompt += (
                "\nYou are also provided with decomposed sub-queries and their search results:\n"
                f"{decomposed_queries_results_str}"
                "Generate a concise and accurate response to the query based on the "
                "decomposed sub-queries and their results. Clearly state your final "
                "answer at the end of the response."
            )
        else:
            prompt += (
                "\nNo supporting API results are available for this query. Answer as "
                "accurately as you can from your own knowledge, say clearly if the "
                "answer cannot be determined, and clearly state your final answer at "
                "the end of the response."
            )
        return prompt
        ### TODO: YOUR CODE ENDS HERE

    def decompose_and_fuse(self, query: str) -> str:
        """
        Decomposes a user query into independent sub-queries for API execution, then synthesizes the resulting outputs into a single, coherent response by prompting multiple models and fusing their generated answers.
        """
        # HINT: Three stages, and the first one is already built above.
        # 1. `decompose_query` gathers the API evidence and `_get_synthesis_prompt` turns it
        #    into a prompt, so there is no new prompt to write for this stage.
        # 2. Sample: send that one synthesis prompt to every model in
        #    `self.fusion_sample_models` and collect their answers. The variety that makes
        #    fusion worthwhile comes from asking different models rather than from varying
        #    the prompt, and `generate` already samples at a non-zero temperature.
        # 3. Fuse: write a second prompt showing the original query alongside those
        #    collected answers, and send it to `self.fusion_model`. Its reply is what this
        #    method returns, and what `evaluate_qa` grades.
        # prompt_help.md has a prompt suggestion for the fusion call.
        ### TODO: YOUR CODE STARTS HERE
        # Stage 1: gather evidence and build the one synthesis prompt.
        decomposed_queries = self.decompose_query(query)
        synthesis_prompt = self._get_synthesis_prompt(query, decomposed_queries)

        # Stage 2: sample one answer per model, in order, skipping failed calls.
        sampled_responses = []
        for sample_model in self.fusion_sample_models:
            sample = self.generate(synthesis_prompt, model=sample_model)
            if sample:
                sampled_responses.append(sample)

        if not sampled_responses:
            # Nothing to fuse. Ask the fusion model the synthesis prompt directly
            # so the pipeline still produces an answer where it can.
            fallback = self.generate(synthesis_prompt, model=self.fusion_model)
            return fallback or "Error: no model produced an answer to the query."

        # Stage 3: fuse the sampled answers with the fusion model.
        responses_str = ""
        for i, sample in enumerate(sampled_responses):
            responses_str += f"Response {i + 1}:\n{sample}\n\n"
        fusion_prompt = (
            f"Your task is to answer the following query: {query}\n"
            "You are also provided with multiple responses to the query:\n"
            f"{responses_str}"
            "Based on the responses, generate a concise and accurate response to the "
            "query. Where the responses disagree, prefer the answer best supported by "
            "the evidence they cite. Clearly state your final answer at the end of the "
            "response."
        )
        fused = self.generate(fusion_prompt, model=self.fusion_model)
        if not fused:
            return sampled_responses[0]
        return fused
        ### TODO: YOUR CODE ENDS HERE

    def _get_iterative_refinement_prompt(self, original_query: str, history: List[Dict[str, Any]]) -> str:
        """
        Helper function to generate the prompt for the iterative refinement model.
        """

        history_str = ""
        for i, item in enumerate(history):
            history_str += f"Step {i+1}:\n"
            history_str += f"Sub-query: {item['sub_query']}\n"
            history_str += f"API: {item['api']}\n"
            history_str += f"Params: {item['params']}\n"
            history_str += f"Results: {_summarize_results(item['results'])}\n"
            if "error" in item:
                history_str += f"Error: {item['error']}\n\n"
            history_str += "\n"

        prompt = f"""
        You are an expert reasoning agent skilled at decomposing complex questions into clear, sequential sub-queries.

        Your objective is to answer the original user query:
        "{original_query}"

        You have access to the following history of sub-queries and their outcomes:
        {history_str if history else "No sub-queries have been executed yet."}

        Your task is to decide the next best action. You can take **exactly one** of the following actions:

        1. **<sub-query>...</sub-query>** — Use this if you need to gather more information before forming the final answer (for example, a previous sub-query failed, returned incomplete data, or raised new questions). The sub-query should be expressed in natural language and represent a single request for information.
        2. **<final_answer>...</final_answer>** — Use this if you have enough information to answer the user's question.

        Guidelines:
        - Output *exactly one* of the above tags — nothing else.
        - Use sub-queries only when *absolutely necessary* for answering the user's query, whether computing intermediate values or gathering additional information.
        - If you do use a sub-query, be concise and direct, and feel free to use intermediate information from the history when specifying the new sub-query (if appropriate).
        - Each sub-query is answered by *one* API call with *one* set of parameters, so ask for exactly one fact at a time: one city, one date, one ticker, or one calculation. A sub-query asking about two cities or three dates will only return the first one.
        - Never do arithmetic yourself. Issue a sub-query for *every* calculation, including simple ones like division or a percentage, and take the number from the result rather than estimating it.
        - Express a calculation as a single straight-forward operation/equation with the numbers substituted in, e.g. "1 / 0.0792006" rather than "the reciprocal of the previous result".
        - Copy dates, tickers, and place names exactly as they appear in the original query. Do not substitute a nearby date.
        """
        return prompt
        

    def iterative_refine(self, query: str, max_iterations: int = 3) -> str:
        """
        Sequentially generates and executes natural language sub-queries to answer a complex query. At each step, the model can either issue a new sub-query (e.g. if additional information is needed, or if a prior sub-query failed), or return the final answer if it has enough information.
        """
        history = []
        for i in range(max_iterations):

            # HINT: Each pass through the loop is two moves: build this step's prompt, then ask
            # the model what to do next. The code below reads the model's choice, runs any
            # sub-query through the API manager, and records it in `history` for the next pass.
            # `_get_iterative_refinement_prompt` above writes the prompt for you; give it the
            # original query, never a sub-query, plus everything gathered in `history` so far --
            # that history is what makes this iterative, since without it every pass would
            # repeat the same first step.
            # Use the refinement model chosen in `__init__` rather than `generate`'s cheaper
            # default: deciding whether to answer or keep digging is the reasoning step here.
            # Leave the reply as raw text, since the tags are parsed below, and consider
            # printing the history each pass -- it is how you see where a wrong answer began,
            # and `_summarize_results` keeps the printed search output readable.
            ### TODO: YOUR CODE STARTS HERE
            print(f"[iterative_refine] iteration {i + 1}/{max_iterations}, history so far:")
            if not history:
                print("  (no sub-queries executed yet)")
            for step_index, item in enumerate(history):
                print(f"  Step {step_index + 1}: {item['sub_query']}")
                print(f"    API: {item['api']} | Params: {item['params']}")
                print(f"    Results: {_summarize_results(item['results'])}")
                if "error" in item:
                    print(f"    Error: {item['error']}")

            prompt = self._get_iterative_refinement_prompt(query, history)
            response = self.generate(prompt, model=self.iterative_refinement_model)
            print(f"[iterative_refine] model response: {response}")
            ### TODO: YOUR CODE ENDS HERE

            if response is None:
                # Spend the remaining iterations retrying rather than aborting.
                continue

            if "<final_answer>" in response:
                # If the model provides a final answer, extract and return it
                final_answer = response.split("<final_answer>")[1].split("</final_answer>")[0].strip()
                return final_answer

            if "<sub-query>" in response:
                # If the model provides a sub-query, extract and execute it
                sub_query = response.split("<sub-query>")[1].split("</sub-query>")[0].strip()
                
                # Use route_query to handle the natural language sub-query
                result = self.api_manager.route_query(sub_query)

                # Add the result of the step to the history for the next iteration
                history.append(_step_result(sub_query, result))
            else:
                # If the model's output is malformed, return an error
                return f"Error: Model did not produce a valid action in iteration {i+1}. Last response: {response}"
        
        # If the loop finishes without a final answer, return an error
        return "Error: Maximum iterations reached without a final answer."

    def run_pipeline(self, query: str) -> str:
        """
        Runs the full agentic pipeline for a given query.
        """
        # Questions spanning three days or two cities need one iteration per API
        # call plus one to answer, and a failed sub-query costs an iteration too.
        # You can use the `max_iterations` parameter to control the maximum number of iterations.
        ### TODO: YOUR CODE STARTS HERE
        # Iterative refinement handles multi-step questions one fact at a time.
        # The budget leaves room for several API calls plus retries of failed ones.
        result = self.iterative_refine(query, max_iterations=10)

        if not result or str(result).startswith("Error:"):
            # The loop ran out of iterations or the model stopped following the
            # tag format. Fall back to parallel decomposition and fusion.
            print(f"[run_pipeline] iterative refinement failed ({result}); falling back to decompose_and_fuse.")
            result = self.decompose_and_fuse(query)

        if not isinstance(result, str) or not result.strip():
            return "Error: the pipeline did not produce an answer."
        return result
        ### TODO: YOUR CODE ENDS HERE