# Ragas Retriever Evaluator Plugin

This plugin provides a tool to evaluate the retriever component of your Cheshire Cat instance using the [Ragas framework](https://docs.ragas.io/). It allows you to measure how well the Cat retrieves relevant context from your documents to answer questions, focusing on core metrics like `Context Precision` and `Context Recall`.

## Key Features

- **On-the-Fly Evaluation**: Run evaluations directly from the chat by providing a link to your dataset.
- **Realistic Benchmarking**: The tool uses the Cat's own memory to retrieve contexts, providing a true-to-life evaluation of your RAG system.
- **Core Retriever Metrics**: Focuses on `Context Precision` and `Context Recall` to measure the quality of retrieved context.
- **Google Sheets Support**: Simply provide a shareable link to a Google Sheet, and the plugin will handle the conversion.
- **Configurable**: Adjust the judge model, temperature, and the number of retrieved documents (`k`) directly from the plugin settings.

## Getting Started

Follow these steps to evaluate your retriever:

### 1. Prepare Your Dataset

Your evaluation data must be in a CSV file or a public Google Sheet. The file must contain three columns: `question`, `ground_truth_contexts`, and `ground_truth`.

For a hands-on example, see the [`dataset-retriever-example.csv`](dataset-retriever-example.csv) file included in this plugin's directory.

| Column Name             | Description                                                                                                                                                    | Example                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `question`              | The question you want to test the retriever against.                                                                                                           | `What is Context Precision?`                                                               |
| `ground_truth_contexts` | A list of strings containing the ideal context chunks needed to answer the question. **The list must be formatted as a string.**                                | `["Context Precision is a metric that evaluates whether the retrieved context is relevant."]` |
| `ground_truth`          | The ideal, complete answer to the question, based *only* on the `ground_truth_contexts`.                                                                       | `Context Precision is a metric that assesses the relevance of the retrieved context.`      |

**Important**:
- The content of `ground_truth_contexts` **must be enclosed in double quotes** and formatted as a valid Python list of strings (e.g., `["context 1", "context 2"]`).
- If using Google Sheets, follow the instructions below to ensure the link is public.

#### Using Google Sheets
To allow the plugin to access your data, you must configure the sharing settings correctly. This is the most common source of errors.

1.  Open your Google Sheet and click the **Share** button (top-right corner).
   ![Example Google Sheet](assets/google-sheet.png)
2.  Under **General access**, change the setting from `Restricted` to **`Anyone with the link`**.
3.  Ensure the role is set to **Viewer**.

![How to share a Google Sheet](assets/google-sheet-sharing.png)

### 2. Configure the Plugin

Before running the evaluation, go to the plugin's settings page in the Cheshire Cat admin panel and configure the following:

- **Plugin Language**: The language for user-facing messages.
- **OpenAI API Key for Judge**: Your OpenAI API key. This is **required** for the judge model to run the evaluation.
- **Judge LLM Model**: The LLM used by Ragas for evaluation (e.g., `gpt-4o`, `gpt-3.5-turbo`).
- **Judge Temperature**: The creativity of the judge model. It's recommended to keep this at `0.0` for consistent results.
- **Retrieval K**: The number of documents (`k`) to retrieve from memory for each question.

![Plugin Settings](assets/ragas-settings.png)

### 3. Run the Evaluation

Once your dataset is ready and the plugin is configured, you can start the evaluation from the chat. Simply ask the Cat to run the evaluation and provide the public URL to your dataset.

**Example Commands:**
```
> Evaluate the retriever using this dataset: https://docs.google.com/spreadsheets/d/your-sheet-id/edit

> run ragas on this: https://example.com/data.csv
```

The plugin will then fetch the dataset, retrieve contexts for each question, and return the evaluation scores directly in the chat.

### 4. Analyze the Results

The tool will output two main components in the chat:

1.  **Evaluation Summary**: A high-level view of the aggregated scores (`context_precision` and `context_recall`) across your entire dataset.
2.  **Detailed Scores per Question**: A Markdown table showing the scores for each individual question, allowing you to quickly spot low-performing queries.

![Evaluation Report](assets/ragas-report.png) 