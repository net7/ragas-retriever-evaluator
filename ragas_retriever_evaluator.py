"""
This plugin provides tools to evaluate the retriever and generator components of a RAG system using the Ragas framework.

It offers two separate evaluations:
1.  **Retriever Evaluation**: Measures the quality of the retrieved context using:
    - `ContextPrecision`: How relevant is the retrieved context?
    - `ContextRecall`: Does the context contain enough information to answer the question?

2.  **Generator Evaluation**: Measures the quality of the generated answer using:
    - `Faithfulness`: Is the answer factually consistent with the context?
    - `ResponseRelevancy`: Is the answer relevant to the question?

Each evaluation is run via a dedicated tool, using a dataset from a public URL (e.g., a Google Sheet).
The plugin reads a dataset file from a public URL (e.g. a Google Sheet link).
For the retriever evaluation, the dataset must contain:
- `question`: The user's query.
- `ground_truth_contexts`: A list of reference chunks (ground truth) for precision.
- `ground_truth`: The ideal answer based on the ground truth contexts, for recall.

For the generator evaluation, the dataset only needs the `question` column.
The configuration for the judge LLM, retrieval parameters, and generation prompt template 
is read from the plugin's settings. The `contexts` are retrieved dynamically by querying 
the Cat's declarative memory for each question, and the `answer` is generated on the fly 
using a configurable prompt template.
"""
import os
import pandas as pd
import ast
import re
from typing import List
import requests
import io
import numpy as np

from cat.mad_hatter.decorators import tool
from cat.looking_glass.stray_cat import StrayCat
from cat.log import log


def get_language(cat: StrayCat) -> str:
    """Gets the user's language from this plugin's settings."""
    try:
        # Attempt to get the language from the plugin's own settings.
        language_setting = cat.mad_hatter.get_plugin().load_settings().get("language")
        # The setting is the full language name (e.g., "Italian").
        if language_setting:
            return language_setting
    except Exception as e:
        log.warning(f"Could not retrieve language setting, defaulting to English. Error: {e}")
    
    return "English" # Default to English if anything goes wrong


def t(text_in_english: str, cat: StrayCat, **kwargs) -> str:
    """
    Translates a given English text into the user's configured language using the LLM.
    If the target language is English, it returns the text directly.
    Variables in curly braces (e.g., {variable}) are preserved.
    """
    lang = get_language(cat)

    # If the target language is English, just format the string and return.
    if lang == "English":
        return text_in_english.format(**kwargs)
    
    try:
        # This prompt asks the LLM to act as a translator.
        # It's instructed to only return the translated text and to keep variables intact.
        prompt = (
            f"You are a professional translator. Translate the following text into {lang}. "
            f"Do not translate any variables enclosed in curly braces (e.g., {{variable}}). "
            f"Only return the translated text, without any additional comments or explanations.\n\n"
            f"Text to translate:\n\"\"\"\n{text_in_english}\n\"\"\""
        )
        
        # Call the LLM to get the translation.
        translated_text = cat.llm(prompt)

        # Clean up the translation by removing potential leading/trailing quotes and whitespace.
        cleaned_text = translated_text.strip().strip('\'"“"')

        # Log the translation for debugging purposes.
        log.info(f"Translated to {lang}: '{cleaned_text}' (original: '{translated_text}')")

        # Format the translated text with any provided variables.
        return cleaned_text.format(**kwargs)
    
    except Exception as e:
        log.error(f"Failed to translate text to {lang}. Error: {e}")
        # Fallback to English if translation fails.
        return text_in_english.format(**kwargs)


def parse_context_list(s: str) -> List[str]:
    """Parses a string representing a list of contexts.

    Uses `ast.literal_eval` to safely convert a string formatted as a
    Python list (e.g., '["chunk 1", "chunk 2"]') into a list object.
    This approach is robust and prevents arbitrary code execution.

    Args:
        s: The input string from a CSV column.

    Returns:
        A list of strings (contexts). Returns an empty list if the
        input is malformed or not a list of strings.
    """
    if not isinstance(s, str) or not s.strip().startswith('[') or not s.strip().endswith(']'):
        return []
    try:
        contexts = ast.literal_eval(s)
        if isinstance(contexts, list) and all(isinstance(item, str) for item in contexts):
            return contexts
        else:
            log.warning(f"Warning: Row '{s}' is not a list of strings and will be ignored.")
            return []
    except (ValueError, SyntaxError):
        log.warning(f"Warning: Could not parse row '{s}'. It will be ignored.")
        return []


@tool(
    return_direct=True,
    examples=[
        # Start evaluation without a URL
        "run a retriever evaluation",
        "evaluate the retriever",
        "start ragas evaluation",
        # Explicit, verbose commands
        "Run a retriever evaluation using this dataset: https://docs.google.com/spreadsheets/d/example/edit",
        "Can you evaluate the retriever with this spreadsheet? https://example.com/data.csv",
        "Evaluate retriever performance with the data at https://example.com/my_test_data.csv",
        # Shorter, more natural commands
        "evaluate this sheet: https://docs.google.com/spreadsheets/d/another-example/edit",
        "run ragas on this: https://example.com/data.csv",
        "evaluate https://example.com/my_test_data.csv"
    ],
)
def evaluate_retriever(dataset_url: str = None, *, cat: StrayCat) -> str:
    """
    Evaluates the retriever's performance using a dataset from a public URL.

    The tool orchestrates the evaluation by:
    1.  Reading a dataset from the provided URL.
    2.  For each question, dynamically retrieving context from the Cat's memory.
    3.  Running Ragas metrics (`ContextPrecision`, `ContextRecall`) to compare the
        retrieved context with the ground truth.
    4.  Displaying the results directly in the chat, including:
        - A summary of the average scores.
        - A detailed table with scores for each question.

    If a URL is not provided, the tool will ask the user for one.

    The dataset must be a publicly accessible CSV file or Google Sheet and MUST contain
    the following columns:
    - `question`: The user's query.
    - `ground_truth_contexts`: A stringified list of ideal context strings.
      Example: '["The first ideal chunk.", "The second ideal chunk."]'
    - `ground_truth`: The ideal answer based on the ground truth contexts.

    Args:
        dataset_url (str, optional): The public URL of the CSV dataset. Defaults to None.
        cat (StrayCat): The Cheshire Cat instance (injected by the framework).
    """
    if not dataset_url:
        return t(
            (
                "**Sure, let's start the retriever evaluation!**\n\n"
                "To proceed, I need you to provide a public link to a CSV file or a Google Sheet. "
                "The file must contain the columns `question`, `ground_truth_contexts`, and `ground_truth`.\n\n"
                "You can paste the link here."
            ),
            cat
        )

    try:
        # MONKEY-PATCH: Prevent nest_asyncio from patching the uvloop, which is
        # the main event loop in Cheshire Cat and is not compatible with nest_asyncio.
        import nest_asyncio
        nest_asyncio.apply = lambda: None

        # Defer imports to avoid import-time side effects with asyncio loops.
        # Ragas internally calls nest_asyncio.apply(), which conflicts with the
        # uvloop used by the Cat. Importing here ensures it runs in the tool's
        # separate thread.
        from datasets import Dataset
        from ragas import evaluate
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
        )

        # Load settings from the plugin's configuration
        settings = cat.mad_hatter.get_plugin().load_settings()
        openai_api_key = settings.get("openai_api_key")
        judge_model = settings.get("judge_model", "gpt-3.5-turbo")
        judge_temperature = float(settings.get("judge_temperature", 0.0))
        k = int(settings.get("retrieval_k", 5))

        if not openai_api_key:
            return "Error: OpenAI API Key for the judge model is not configured. Please go to the Ragas Retriever Evaluator plugin settings and add your key."

        # Set the API key as an environment variable for Ragas and LangChain to find.
        # This is the most reliable way to ensure all underlying components get the key.
        os.environ["OPENAI_API_KEY"] = openai_api_key

        cat.send_ws_message(f"Starting RETRIEVER evaluation with Ragas...\nDataset: {dataset_url}\nJudge Model: {judge_model}, k: {k}", "notification")

        # Handle dataset location (URL or local path)
        path_or_url = dataset_url
        if not path_or_url.startswith("http"):
             return "Error: Please provide a valid public URL, not a local path."

        # Check if it's a Google Sheet URL and convert it if it's a standard shareable link
        if "docs.google.com/spreadsheets" in path_or_url:
            match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)/edit", path_or_url)
            if match:
                sheet_id = match.group(1)
                path_or_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                log.info(f"Standard Google Sheet URL detected. Converting to export URL: {path_or_url}")
            else:
                # Assume other Google Sheet URLs (like /pub) are already direct download links
                log.info("Published or other Google Sheet URL detected. Using as is.")

        # Read and validate the dataset
        try:
            # Use requests to fetch the content with a user-agent to avoid potential access issues.
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            }
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

            # Read the CSV data from the response content, explicitly decoding as UTF-8
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Provide user-friendly instructions for the most common error (private sheet)
                error_message_en = (
                    "**Error: Access Denied to Google Sheet (401 Unauthorized)**\n\n"
                    "It seems the file is not public. To fix this, please change the sharing settings:\n\n"
                    "1.  **Open your Google Sheet**.\n"
                    "2.  Click on **Share** (top right).\n"
                    "3.  Under **General access**, change from `Restricted` to **`Anyone with the link`**.\n"
                    "4.  Ensure the role is **Viewer**.\n\n"
                    "After updating the settings, please try sending the command again."
                )
                error_message = t(error_message_en, cat)
                log.error(f"Access denied to Google Sheet at {dataset_url} (401). It is likely not public.")
            elif e.response.status_code == 404:
                error_message_en = "**Error: File Not Found (404 Not Found)**\n\n" \
                                   "The provided URL was not found: `{dataset_url}`. Please check that it is correct and try again."
                error_message = t(error_message_en, cat, dataset_url=dataset_url)
                log.error(f"Dataset not found at {dataset_url} (404).")
            else:
                error_message_en = "Error: Could not read the dataset from '{dataset_url}'. The server returned an error: {e}. Please ensure the URL is correct and publicly accessible."
                error_message = t(error_message_en, cat, dataset_url=dataset_url, e=e)
                log.error(f"HTTP Error when fetching dataset from {dataset_url}: {e}")
            
            return error_message
        except Exception as e:
            error_message = f"Error: Could not process the dataset from '{dataset_url}'. Please check the URL and file format. Details: {e}"
            log.error(error_message)
            return error_message

        # Basic validation
        required_columns = ["question", "ground_truth_contexts", "ground_truth"]
        if not all(col in df.columns for col in required_columns):
            error_message = f"Error: The dataset must contain the following columns: {', '.join(required_columns)}."
            log.error(error_message)
            return error_message

        df.dropna(subset=["question"], inplace=True)
        
        # Dynamically retrieve contexts from the Cat's memory
        retrieved_contexts = []
        for idx, row in df.iterrows():
            question = row["question"]
            log.info(f"Retrieving context for question: '{question}'")
            try:
                # 1. Embed the query
                embedding = cat.embedder.embed_query(question)
                
                # 2. Recall documents from declarative memory
                recalled_docs = cat.memory.vectors.declarative.recall_memories_from_embedding(embedding, k=k)
                
                # 3. Extract page_content
                contexts = [doc[0].page_content for doc in recalled_docs]
                retrieved_contexts.append(contexts)
            except Exception as e:
                log.error(f"Failed to retrieve context for question '{question}': {e}")
                retrieved_contexts.append([]) # Append empty list on error

        df["contexts"] = retrieved_contexts
        
        # Parse ground truth columns
        df["ground_truth_contexts"] = df["ground_truth_contexts"].apply(parse_context_list)

        # Prepare dataset for Ragas
        dataset = Dataset.from_pandas(df)

        # Set up Ragas metrics
        langchain_llm = ChatOpenAI(
            model=judge_model,
            temperature=judge_temperature
        )
        judging_llm = LangchainLLMWrapper(langchain_llm)

        metrics_to_evaluate = [
            ContextPrecision(),
            ContextRecall(llm=judging_llm),
        ]

        cat.send_ws_message("Running evaluation...", "notification")
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_evaluate,
        )
        cat.send_ws_message("Evaluation completed.", "notification")

        # The result object contains the scores. We convert it to a DataFrame.
        result_scores_df = result.to_pandas()

        # To ensure we have all original columns (like 'question'), we concatenate
        # our input DataFrame (`df`) with the new scores DataFrame.
        # They are guaranteed to be in the same order.
        df.reset_index(drop=True, inplace=True)
        result_scores_df.reset_index(drop=True, inplace=True)
        detailed_df = pd.concat([df, result_scores_df], axis=1)

        # The result object contains the summary scores as lists. We need to average them.
        # We can format them into a more readable markdown list.
        summary_lines = [
            f"- **Context Precision:** {np.nanmean(result['context_precision']):.3f}",
            f"- **Context Recall:** {np.nanmean(result['context_recall']):.3f}"
        ]
        summary_str = "\n".join(summary_lines)

        log.info(f"Ragas Evaluation Results (Summary):\n{summary_str}")

        # --- Create Markdown table for chat display ---
        # Select, rename, and format columns for a clean table
        display_df = detailed_df[["question", "context_precision", "context_recall"]].copy()
        display_df.rename(columns={
            "question": "Question",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall"
        }, inplace=True)
        
        # Truncate long questions to avoid a very wide table in the chat
        display_df["Question"] = display_df["Question"].apply(lambda x: (x[:50] + '...') if len(x) > 53 else x)
        
        detailed_markdown_table = display_df.to_markdown(index=False, floatfmt=".3f")
        # --- End of Markdown table creation ---

        # Log the full details for debugging, without saving to a file
        log.info(f"Detailed Ragas evaluation report (not saved):\n{detailed_df.to_string()}")

        # Translate components separately for better formatting control.
        header = t("**Retriever Evaluation Summary:**", cat)
        subheader = t("### Detailed Scores per Question:", cat)
        
        # Assemble the final message, ensuring proper spacing.
        final_message = f"{header}\n\n{summary_str}\n\n{subheader}\n{detailed_markdown_table}"
        
        return final_message

    except Exception as e:
        log.error(f"An unexpected error occurred during retriever evaluation: {e}")
        return f"An error occurred during evaluation. Check the logs for details. Error: {e}" 


@tool(
    return_direct=True,
    examples=[
        # Start evaluation without a URL
        "run a generator evaluation",
        "evaluate the generator",
        "start ragas generator evaluation",
        # Explicit, verbose commands
        "Run a generator evaluation using this dataset: https://docs.google.com/spreadsheets/d/example/edit",
        "Can you evaluate the generator with this spreadsheet? https://example.com/data.csv",
        # Shorter, more natural commands
        "evaluate generator with this sheet: https://docs.google.com/spreadsheets/d/another-example/edit",
        "run ragas generator on this: https://example.com/data.csv",
    ],
)
def evaluate_generator(dataset_url: str = None, *, cat: StrayCat) -> str:
    """
    Evaluates the generator's performance using a dataset from a public URL.

    This tool focuses on the generation part of the RAG pipeline, measuring:
    - `Faithfulness`: Whether the answer is factually consistent with the retrieved context.
    - `ResponseRelevancy`: Whether the answer is relevant to the question.

    The tool orchestrates the evaluation by:
    1.  Reading a dataset with `question`s from the provided URL.
    2.  For each question:
        a. Dynamically retrieving context from the Cat's memory.
        b. Generating an answer based on the retrieved context.
    3.  Running Ragas metrics to evaluate the generated answers.
    4.  Displaying the results directly in the chat.

    If a URL is not provided, the tool will ask the user for one.

    The dataset must be a publicly accessible CSV file or Google Sheet and MUST contain
    the following column:
    - `question`: The user's query.

    Args:
        dataset_url (str, optional): The public URL of the CSV dataset. Defaults to None.
        cat (StrayCat): The Cheshire Cat instance (injected by the framework).
    """
    if not dataset_url:
        return t(
            (
                "**Sure, let's start the generator evaluation!**\n\n"
                "To proceed, I need you to provide a public link to a CSV file or a Google Sheet. "
                "The file must contain the column `question`.\n\n"
                "You can paste the link here."
            ),
            cat
        )

    try:
        # MONKEY-PATCH: Prevent nest_asyncio from patching the uvloop.
        import nest_asyncio
        nest_asyncio.apply = lambda: None

        # Defer imports to avoid import-time side effects with asyncio loops.
        from datasets import Dataset
        from ragas import evaluate
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
        )

        # Load settings from the plugin's configuration
        settings = cat.mad_hatter.get_plugin().load_settings()
        openai_api_key = settings.get("openai_api_key")
        judge_model = settings.get("judge_model", "gpt-3.5-turbo")
        judge_temperature = float(settings.get("judge_temperature", 0.0))
        k = int(settings.get("retrieval_k", 5))
        prompt_template = settings.get("generation_prompt_template", 
            "Based *only* on the following context, answer the question. Do not use any other information.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")

        if not openai_api_key:
            return "Error: OpenAI API Key for the judge model is not configured. Please go to the Ragas Retriever Evaluator plugin settings and add your key."

        os.environ["OPENAI_API_KEY"] = openai_api_key

        cat.send_ws_message(f"Starting GENERATOR evaluation with Ragas...\nDataset: {dataset_url}\nJudge Model: {judge_model}, k: {k}", "notification")

        path_or_url = dataset_url
        if not path_or_url.startswith("http"):
             return "Error: Please provide a valid public URL, not a local path."

        if "docs.google.com/spreadsheets" in path_or_url:
            match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)/edit", path_or_url)
            if match:
                sheet_id = match.group(1)
                path_or_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                log.info(f"Standard Google Sheet URL detected. Converting to export URL: {path_or_url}")
            else:
                log.info("Published or other Google Sheet URL detected. Using as is.")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            }
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_message_en = (
                    "**Error: Access Denied to Google Sheet (401 Unauthorized)**\n\n"
                    "It seems the file is not public. To fix this, please change the sharing settings:\n\n"
                    "1.  **Open your Google Sheet**.\n"
                    "2.  Click on **Share** (top right).\n"
                    "3.  Under **General access**, change from `Restricted` to **`Anyone with the link`**.\n"
                    "4.  Ensure the role is **Viewer**.\n\n"
                    "After updating the settings, please try sending the command again."
                )
                error_message = t(error_message_en, cat)
                log.error(f"Access denied to Google Sheet at {dataset_url} (401). It is likely not public.")
            elif e.response.status_code == 404:
                error_message_en = "**Error: File Not Found (404 Not Found)**\n\n" \
                                   "The provided URL was not found: `{dataset_url}`. Please check that it is correct and try again."
                error_message = t(error_message_en, cat, dataset_url=dataset_url)
                log.error(f"Dataset not found at {dataset_url} (404).")
            else:
                error_message_en = "Error: Could not read the dataset from '{dataset_url}'. The server returned an error: {e}. Please ensure the URL is correct and publicly accessible."
                error_message = t(error_message_en, cat, dataset_url=dataset_url, e=e)
                log.error(f"HTTP Error when fetching dataset from {dataset_url}: {e}")
            
            return error_message
        except Exception as e:
            error_message = f"Error: Could not process the dataset from '{dataset_url}'. Please check the URL and file format. Details: {e}"
            log.error(error_message)
            return error_message

        # Basic validation
        required_columns = ["question"]
        if not all(col in df.columns for col in required_columns):
            error_message = f"Error: The dataset must contain the following column: {', '.join(required_columns)}."
            log.error(error_message)
            return error_message

        df.dropna(subset=["question"], inplace=True)
        
        # Dynamically retrieve contexts and generate answers
        retrieved_contexts = []
        generated_answers = []
        for idx, row in df.iterrows():
            question = row["question"]
            log.info(f"Processing question for generator evaluation: '{question}'")
            try:
                # 1. Retrieve contexts from memory
                embedding = cat.embedder.embed_query(question)
                recalled_docs = cat.memory.vectors.declarative.recall_memories_from_embedding(embedding, k=k)
                contexts = [doc[0].page_content for doc in recalled_docs]
                retrieved_contexts.append(contexts)

                # 2. Generate an answer based on the contexts
                context_str = "\n".join(contexts)
                prompt = prompt_template.format(context=context_str, question=question)
                answer = cat.llm(prompt)
                generated_answers.append(answer)

            except Exception as e:
                log.error(f"Failed to retrieve context or generate answer for question '{question}': {e}")
                retrieved_contexts.append([])
                generated_answers.append("") # Append empty strings on error

        df["contexts"] = retrieved_contexts
        df["answer"] = generated_answers

        # Prepare dataset for Ragas
        dataset = Dataset.from_pandas(df)

        # Set up Ragas metrics
        langchain_llm = ChatOpenAI(
            model=judge_model,
            temperature=judge_temperature
        )
        judging_llm = LangchainLLMWrapper(langchain_llm)

        # Wrap the Cat's embedder for Ragas
        embedder = LangchainEmbeddingsWrapper(cat.embedder)

        metrics_to_evaluate = [
            Faithfulness(llm=judging_llm),
            ResponseRelevancy(llm=judging_llm, embeddings=embedder),
        ]

        cat.send_ws_message("Running evaluation...", "notification")
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_evaluate,
        )
        cat.send_ws_message("Evaluation completed.", "notification")

        result_scores_df = result.to_pandas()
        df.reset_index(drop=True, inplace=True)
        result_scores_df.reset_index(drop=True, inplace=True)
        detailed_df = pd.concat([df, result_scores_df], axis=1)

        summary_lines = [
            f"- **Faithfulness:** {np.nanmean(result['faithfulness']):.3f}",
            f"- **Answer Relevancy:** {np.nanmean(result['answer_relevancy']):.3f}"
        ]
        summary_str = "\n".join(summary_lines)

        log.info(f"Ragas Generator Evaluation Results (Summary):\n{summary_str}")

        # --- Create Markdown table for chat display ---
        display_df = detailed_df[["question", "faithfulness", "answer_relevancy"]].copy()
        display_df.rename(columns={
            "question": "Question",
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy"
        }, inplace=True)
        
        display_df["Question"] = display_df["Question"].apply(lambda x: (x[:50] + '...') if len(x) > 53 else x)
        
        detailed_markdown_table = display_df.to_markdown(index=False, floatfmt=".3f")

        log.info(f"Detailed Ragas generator evaluation report (not saved):\n{detailed_df.to_string()}")

        header = t("**Generator Evaluation Summary:**", cat)
        subheader = t("### Detailed Scores per Question:", cat)
        
        final_message = f"{header}\n\n{summary_str}\n\n{subheader}\n{detailed_markdown_table}"
        
        return final_message

    except Exception as e:
        log.error(f"An unexpected error occurred during generator evaluation: {e}")
        return f"An error occurred during evaluation. Check the logs for details. Error: {e}" 