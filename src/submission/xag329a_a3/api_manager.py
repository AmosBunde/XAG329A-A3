import requests
import json
from pathlib import Path
from tavily import TavilyClient
from geopy.geocoders import Nominatim
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from massive import RESTClient
import xag329a_a3.wolfram_api as wolfram_api
import nest_asyncio
from xag329a_a3.utils import generate_structured

# Pydantic models for structured LLM parsing
class TavilySearchParams(BaseModel):
    search_query: str
    num_results: int = Field(default=2, description="The number of results to return from the Tavily search")

class StockDataParams(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to get data for, e.g. 'AAPL'")
    date: str = Field(description="The date to get stock data for, in the format 'YYYY-MM-DD'")

class ComputeParams(BaseModel):
    # Examples are deliberately unrelated to the eval questions: models copy
    # numbers and bounds out of this description when the query omits them.
    wolfram_query: str = Field(description="The mathematical computation / operation to perform, as a plain expression taking all numbers and bounds from the user's query, e.g. 'integrate e^(-x^2) from 0 to 3' or 'solve 2*y + 9 = 31 for y'. Do not add rounding or formatting instructions such as 'to the nearest cent', and do not use Mathematica syntax such as 'NSolve[...]'.")

class GetWeatherParams(BaseModel):
    location: str = Field(description="The location to get weather data for, in the format 'City, Country'")
    date: str = Field(description="The date to get weather data for, in the format 'YYYY-MM-DD'")
    hour: str = Field(default="12", description="The hour of the day to get weather data for, in the format 'HH' (24-hour format)")

class APIResponse(BaseModel):
    api_name: str = Field(description="The name of the API to use for the query. Must be one of: 'tavily_search', 'get_stock_data', 'compute', 'get_weather'")


def _api_error(result: Any) -> Optional[str]:
    """Returns the error message from an API result, or None if it succeeded.

    Tools return either a dict or, in the case of `tavily_search`, a list of
    dicts, so both shapes are checked for an "error" key.
    """
    if isinstance(result, dict):
        return result.get("error") or None
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("error"):
                return item["error"]
    return None


# The notebook sits beside this package next to weather_codes.json, but resolving
# the path here lets `APIManager` be built from any working directory.
_WEATHER_CODES_PATH = Path(__file__).resolve().parent.parent / "weather_codes.json"


class APIManager:
    """A unified class to manage various API interactions, using an LLM to route queries."""
    
    def __init__(
        self, 
        tavily_api_key: str,
        polygon_api_key: str, 
        wolfram_app_id: str, 
        router_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct"
    ):
        """Initializes the APIManager with the necessary API keys."""
        
        with open(_WEATHER_CODES_PATH, "r") as f:
            self._weather_codes = json.load(f)

        self.api_functions = {
            "tavily_search": (self.tavily_search, TavilySearchParams), 
            "get_stock_data": (self.get_stock_data, StockDataParams),
            "compute": (self.compute, ComputeParams),
            "get_weather": (self.get_weather, GetWeatherParams),
        }

        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.search_truncate = 512
        self.polygon_client = RESTClient(api_key=polygon_api_key)
        self.wolfram_client = wolfram_api.Client(wolfram_app_id)
        self.geolocator = Nominatim(user_agent="xag329a-a3")
        self.router_model = router_model

    def _parse_query_params(
        self,
        query: str, 
        function_name: str,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Uses an LLM to parse parameters for a given function from a user query.

        Args:
            query (str): The natural language user query.
            function_name (str): The name of the API function to parse parameters for.
            model (Optional[str]): The specific model to use for parsing. Defaults to the
                instance's router_model.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of validated parameters, or None if
            parsing or validation fails.
        """

        # HINT: `self.api_functions` pairs each tool with the Pydantic model describing
        # its arguments. `generate_structured` in `utils.py` asks the model for JSON and
        # validates it against that Pydantic model in one call
        # (https://openrouter.ai/docs/guides/features/structured-outputs), so the work
        # here is the prompt rather than the parsing.
        # Tell the model which function it is filling in, what that function does, and
        # the JSON shape to answer with -- the tool's own docstring and its Pydantic
        # schema both describe this -- and say explicitly to reply with JSON only.
        # This method hands back a plain dict, not the validated object.
        # prompt_help.md has a prompt suggestion
        ### TODO: YOUR CODE STARTS HERE
        if function_name not in self.api_functions:
            print(f"Unknown function '{function_name}'; cannot parse parameters.")
            return None
        api_func, schema_model = self.api_functions[function_name]
        if schema_model is None:
            print(f"No parameter schema registered for '{function_name}'.")
            return None

        function_doc = " ".join((api_func.__doc__ or "").split())
        system_prompt = (
            f"Extract the parameters for the function '{function_name}' from the user query. "
            f"The function documentation is: '{function_doc}'. "
            "Respond ONLY with a valid JSON object that adheres to the provided schema:\n"
            f"{schema_model.model_json_schema()}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        parsed = generate_structured(model or self.router_model, messages, schema_model)
        if parsed is None:
            return None
        return parsed.model_dump()
        ### TODO: YOUR CODE ENDS HERE

    def route_query(self, query: str, model: Optional[str] = None) -> Dict:
        """Determines the appropriate API for a query, parses parameters, and executes the API call.

        This method first uses an LLM to decide which API tool is best suited for the
        user's query. It then calls `parse_query_params` to extract the necessary
        arguments and finally executes the chosen API function.

        Args:
            query (str): The natural language user query.
            model (Optional[str]): The specific model to use for routing. Defaults to the
                instance's router_model.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the API call, the
            name of the API used, the parameters, and any potential errors.
        """

        # HINT: Routing takes two model-driven steps, and the code after this block
        # calls the tool for you.
        # 1. Let the model pick a tool. `generate_structured` with the `APIResponse`
        #    model hands the decision back already parsed, and the tool names to offer
        #    the model are the keys of `self.api_functions`.
        # 2. Get that tool's arguments by reusing `_parse_query_params` above, rather
        #    than writing a second structured call here.
        # Leave the chosen tool name in `function_name` and its arguments in `params`,
        # which is what the code below expects.
        # prompt_help.md has a prompt suggestion for the routing call.
        ### TODO: YOUR CODE STARTS HERE
        routing_model = model or self.router_model

        # One sentence per tool saying when it applies. Any tool without an entry
        # here falls back to its own docstring.
        tool_usage = {
            "compute": (
                "Use for any arithmetic, unit conversion, percentage, average, "
                "or other mathematical calculation."
            ),
            "get_weather": (
                "Use for the weather, temperature, humidity, or wind at a specific "
                "place on a specific date and hour."
            ),
            "get_stock_data": (
                "Use for the open, high, low, close, or volume of a stock ticker "
                "on a specific date."
            ),
            "tavily_search": (
                "Use for anything else: general knowledge, facts, recent events, "
                "or information the other tools do not cover."
            ),
        }
        tool_lines = []
        for name, (api_func, _) in self.api_functions.items():
            usage = tool_usage.get(name) or " ".join((api_func.__doc__ or "").split())
            tool_lines.append(f"- {name}: {usage}")

        system_prompt = (
            "Based on the user's query, determine the most appropriate tool to use. "
            "Respond ONLY with a valid JSON object that adheres to the provided schema:\n"
            f"{APIResponse.model_json_schema()}\n\n"
            f"Available Tools: {', '.join(self.api_functions.keys())}\n\n"
            "When to use each tool:\n" + "\n".join(tool_lines)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        routing = generate_structured(routing_model, messages, APIResponse)
        if routing is None:
            return {
                "results": None,
                "api_used": None,
                "params": None,
                "error": "Routing failed: the model did not choose a tool.",
            }

        function_name = routing.api_name
        if function_name not in self.api_functions:
            return {
                "results": None,
                "api_used": function_name,
                "params": None,
                "error": f"Routing failed: '{function_name}' is not an available tool.",
            }

        params = self._parse_query_params(query, function_name, model=model)
        if params is None:
            return {
                "results": None,
                "api_used": function_name,
                "params": None,
                "error": f"Parameter parsing failed for '{function_name}'.",
            }
        ### TODO: YOUR CODE ENDS HERE

        # Call the API function
        try:
            api_func, _ = self.api_functions[function_name]
            result = api_func(**params)
            # If there is an error in the result of the API call, return the error
            error = _api_error(result)
            if error:
                return {"results": None, "api_used": function_name, "params": params, "error": error}
            return {"results": result, "api_used": function_name, "params": params}
        except Exception as e:
            # If there is an error from the API call itself, return the error
            return {"results": None, "api_used": function_name, "params": params, "error": str(e)}

    def tavily_search(self, search_query: str, num_results: int = 2) -> List[Dict]:
        """Performs a Tavily search for general knowledge questions, recent events, or information not covered by other tools."""
        
        # This function should return a list of dictionaries, where each dictionary contains the `title`, `url`, `webpage_content` (use `content`), and `snippet` (truncated content) of a search result.
        # For content usage, you may use `snippet` as truncated content or truncate `webpage_content` as needed.
        # If there is an error, return a list with a dictionary containing an "error" key containing the error message.

        # HINT: Use the Tavily Search API to perform a search for the given query.
        # API test notebook: [tavily.ipynb](./eval-search-apis/tavily.ipynb)
        # https://docs.tavily.com/sdk/python/quick-start
        # https://docs.tavily.com/sdk/python/reference
        try:
            response = self.tavily_client.search(search_query, max_results=num_results)
            results = []
            # `num_results` is an upper bound: Tavily returns fewer pages for
            # narrow queries and caps the count regardless of what is asked.
            pages = response['results']
            if not pages:
                return [{"error": f"No search results found for '{search_query}'."}]
            for page in pages:
                content = page['content']
                snippet = content
                if len(content) > self.search_truncate:
                    snippet = content[:self.search_truncate]
                    end = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
                    if end != -1:
                        snippet = snippet[: end + 1]
                    snippet = snippet.rstrip()
                results.append({'title': page['title'], 'url':page['url'], 'snippet': snippet, 'webpage_content':content})
            return results
        except Exception as e:
            return [{"error": f"Error during search: {e}"}]

        
    def get_stock_data(self, ticker: str, date: str) -> Dict[str, Any]:
        """Retrieves historical stock data (open, high, low, close, volume) for a given ticker (e.g. 'AAPL') and YYYY-MM-DD date (e.g. '2025-03-05')."""
        
        # This function should return a dictionary containing the date, open, high, low, close, and volume of the stock data.
        # If there is an error, return a dictionary with an "error" key containing the error message.
        # HINT: API test notebook: [polygon.ipynb](./eval-search-apis/polygon.ipynb)

        try:
            result = self.polygon_client.get_daily_open_close_agg(ticker=ticker, date=date)
            result_dict = dict(
                date=result.from_,
                open=result.open,
                high=result.high,
                low=result.low,
                close=result.close,
                volume=result.volume
            )
        except Exception as e:
            result_dict = {"error": f"Error fetching stock data for {ticker}: {e}"}
        return result_dict
        
    def get_weather(self, location: str, date: str, hour: str = "12") -> Dict[str, Any]:
        """
        Fetches weather data (temperature, humidity, wind speed) for a specific location (e.g. 'Palo Alto, CA') and YYYY-MM-DD date (e.g. '2025-10-22') and HH hour (e.g. '15' for 3PM).
        """
        
        # This function should return a dictionary containing the temperature, weather description, humidity, and wind speed of the weather data.
        # If there is an error, return a dictionary with an "error" key containing the error message.
        # HINT: Use the Nominatim geocoder to get the latitude and longitude of the location, 
        # and then use the Open Meteo API to fetch weather data for the given location, date, and hour.
        # You can translate a numerical weather code to a weather description using the `_weather_codes` dictionary.
        try:
            geocoded = self.geolocator.geocode(location)
            if not geocoded:
                return {"error": "Location not found"}
            params = {
                "latitude": geocoded.latitude,
                "longitude": geocoded.longitude,
                "start_date": date,
                "end_date": date,
                "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit",
            }

            weather_data = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params).json()

            if "hourly" not in weather_data:
                return {"error": "No hourly data found"}
            data = weather_data["hourly"]
            try:
                idx = data["time"].index(f"{date}T{str(hour).zfill(2)}:00")
            except ValueError:
                return {"error": "Hour not found in the data"}
            wmo_code = data["weather_code"][idx]

            temperature = data["temperature_2m"][idx]
            humidity = data["relative_humidity_2m"][idx]
            wind_speed = data["wind_speed_10m"][idx]
            weather_description = self._weather_codes.get(str(wmo_code), "Unknown")

            result_dict = dict(
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                weather_description=weather_description,
            )
        except Exception as e:
            result_dict = {"error": f"Error fetching weather data for {location} on {date} at {hour}: {e}"}
        return result_dict

    def compute(self, wolfram_query: str) -> Dict[str, str]:
        """Performs a mathematical computation or operation using Wolfram Alpha."""
        
        # This function should return a dictionary containing the `results` of the computation.
        # If there is an error, return a dictionary with an "error" key containing the error message.
        try:
            nest_asyncio.apply()
            # HINT: Use the `answer` function from `xag329a_a3/wolfram_api.py`, which queries
            # Wolfram|Alpha and pulls the answer out of the response.
            # HINT: API test notebook: [wolframalpha.ipynb](./eval-search-apis/wolframalpha.ipynb)
            return {"result": wolfram_api.answer(self.wolfram_client, wolfram_query)}
        except ValueError as e:
            detail = str(e)
        except Exception as e:
            # httpx.ReadTimeout stringifies to "", so name the exception type
            # explicitly rather than printing a blank message.
            detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        message = f"Error fetching Wolfram Alpha data for {wolfram_query}: {detail}"
        print(message)
        return {"error": message}
