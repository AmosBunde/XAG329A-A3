# Prompt suggestion


**NOTE**: This note is provided for your reference. Some prompt examples omit variables on purpose, so you need to find and fill them in.

## AIPManger._parse_query_params()


This is a prompt to parse the user query:

```
        system_prompt = (
            f"Extract the parameters for the function '{...}' from the user query. "
            f"The function documentation is: '{...}'. "
            f"Respond ONLY with a valid JSON object that adheres to the provided schema:\n{...}"
        )
```

## AIPManager.route_query

```
        system_prompt = f"Based on the user's query, determine the most appropriate tool to use. Respond ONLY with a valid JSON object that adheres to the provided schema:\n{APIResponse.model_json_schema()}\n\nAvailable Tools: {', '.join(self.api_functions.keys())}"
```


## MultiLMAgent.single_LM_with_single_API_call

```
        prompt = "Here is the query: " + query + "\nHere is the response from the API: " + str(results["results"]) + "\nGenerate a concise and accurate response to the query based on the API response. Clearly state your final answer at the end of the response."
```


## MultiLMAgent._get_query_decomposition_prompt

```
        prompt = f"""
        Your task is to break down the following user query into the minimal set of sub-queries needed for the API manager to answer it effectively.

        User query:
        {...}

        Available API functions:
        {...}

        Guidelines:
        - Each sub-query should be enclosed in <sub-query>...</sub-query> tags.
        - Sub-queries should be distinct and complementary, covering different aspects of the user's request.
        - Only include as many sub-queries as are truly necessary to fully answer the user query.
        - Sub-queries should NOT be dependent on each other; they will be executed independently, in parallel.
        - The sub-queries should be able to be answered without relying on the results of other sub-queries.
        """
```


## MultiLMAgent._get_synthesis_prompt
```
            prompt += f"\nYou are also provided with decomposed sub-queries and their search results:\n{decomposed_queries_results_str}\nGenerate a response to the query based on the decomposed sub-queries."

```

## MultiLMAgent.decompose_and_fuse

```
        fusion_prompt = f"Your task is to answer the following query: {query}\nYou are also provided with multiple responses to the query:\n{responses_str}\nBased on the responses, generate a concise and accurate response to the query."
```