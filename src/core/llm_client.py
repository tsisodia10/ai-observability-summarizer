"""
LLM Client and Prompt Building Functions

Contains all business logic for interacting with LLMs (local and external),
building prompts, and processing LLM responses.
"""

import re
import requests
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Any
from .config import CHAT_SCOPE_FLEET_WIDE
from datetime import datetime, timedelta, timezone, time
from dateparser.search import search_dates

from .config import MODEL_CONFIG, LLM_API_TOKEN, LLAMA_STACK_URL, VERIFY_SSL

import logging
from common.pylogger import get_python_logger

# Initialize structured logger once - other modules should use logging.getLogger(__name__)
get_python_logger()

logger = logging.getLogger(__name__)
from .response_validator import ResponseValidator, ResponseType

# LLM Generation Configuration Constants
DETERMINISTIC_TEMPERATURE = 0  # Zero temperature for consistent, deterministic output
DEFAULT_MAX_TOKENS = 6000  # Maximum tokens allowed
DEFAULT_SSL_VERIFICATION = True  # Enable SSL verification for external API calls

# Time Range Configuration Constants
DEFAULT_TIME_RANGE_HOURS = 1  # Default lookback period when no time specified
DEFAULT_RATE_SYNTAX = "1h"  # Default PromQL time range syntax
FALLBACK_RATE_SYNTAX = "5m"  # Fallback PromQL time range for shorter queries


def _make_api_request(
    url: str, headers: dict, payload: dict, verify_ssl: bool = True
) -> dict:
    """Make API request with consistent error handling"""
    response = requests.post(url, headers=headers, json=payload, verify=verify_ssl)
    response.raise_for_status()
    return response.json()


def _validate_and_extract_response(
    response_json: dict, is_external: bool, provider: str = "LLM"
) -> str:
    """Validate response format and extract content"""
    if is_external:
        if provider == "google":
            # Google Gemini response format
            if "candidates" not in response_json or not response_json["candidates"]:
                raise ValueError(f"Invalid {provider} response format")

            candidate = response_json["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise ValueError(f"Invalid {provider} response structure")

            parts = candidate["content"]["parts"]
            if not parts or "text" not in parts[0]:
                raise ValueError(f"Invalid {provider} response content")

            return parts[0]["text"].strip()
        else:
            # OpenAI and other providers using "choices" format
            if "choices" not in response_json or not response_json["choices"]:
                raise ValueError(f"Invalid {provider} response format")

            return response_json["choices"][0]["message"]["content"].strip()
    else:
        # Local model response format
        if "choices" not in response_json or not response_json["choices"]:
            raise ValueError(f"Invalid {provider} response format")
        return response_json["choices"][0]["text"].strip()


def _clean_llm_summary_string(text: str) -> str:
    """Remove non-printable ASCII characters and normalize whitespace"""
    # Remove any non-printable ASCII characters (except common whitespace like space, tab, newline)
    cleaned_text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
    # Replace multiple spaces/newlines/tabs with single spaces, then strip leading/trailing whitespace
    return re.sub(r"\s+", " ", cleaned_text).strip()


def _load_mlops_metrics_config() -> Dict[str, Any]:
    """Load MLOps metrics configuration from JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "mlops_metrics_config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        # Fallback to basic metrics if config file not found
        return {
            "mlops_critical_metrics": {
                "user_experience": {
                    "metrics": {
                        "P95 Latency (s)": {"priority": 1},
                        "Requests Running": {"priority": 1}
                    }
                }
            }
        }


def _get_priority_metrics(metric_dfs: Dict[str, Any], max_metrics: int = 8) -> Dict[str, Any]:
    """Filter to only the most important metrics for MLOps analysis"""
    
    config = _load_mlops_metrics_config()
    
    # Create priority mapping from config
    priority_metrics = {}
    
    for category, category_data in config["mlops_critical_metrics"].items():
        category_priority = category_data.get("priority", 5)
        for metric_name, metric_config in category_data["metrics"].items():
            if metric_name in metric_dfs:
                priority_metrics[metric_name] = {
                    "priority": category_priority,
                    "config": metric_config,
                    "df": metric_dfs[metric_name]
                }
    
    # Sort by priority and return top metrics
    sorted_metrics = sorted(priority_metrics.items(), key=lambda x: x[1]["priority"])
    
    result = {}
    for metric_name, metric_info in sorted_metrics[:max_metrics]:
        result[metric_name] = metric_info["df"]
    
    return result


def _add_simple_intelligence(metric_name: str, df) -> str:
    """Add statistical intelligence with contextual assessment - Phase 2.1 enhancement"""
    
    if df is None or df.empty or len(df) < 3:
        return ""
    
    try:
        values = df['value'].values
        current_value = values[-1]
        insights = []
        
        # Load thresholds from configuration
        config = _load_mlops_metrics_config()
        metric_config = None
        
        # Find metric configuration across all categories
        for category_data in config["mlops_critical_metrics"].values():
            if metric_name in category_data["metrics"]:
                metric_config = category_data["metrics"][metric_name]
                break
        
        # Phase 1: Volatility analysis using coefficient of variation
        if len(values) > 1 and np.mean(values) > 0:
            cv = (np.std(values) / np.mean(values)) * 100
            if cv > 20:
                insights.append(f"🌪️ High volatility ({cv:.1f}%)")
            elif cv > 10:
                insights.append(f"📊 Moderate volatility ({cv:.1f}%)")
            else:
                insights.append(f"🎯 Stable ({cv:.1f}%)")
        
        # Phase 2.1: Data-driven contextual assessment using percentiles
        if len(values) >= 5:
            p25 = np.percentile(values, 25)
            p75 = np.percentile(values, 75)
            p95 = np.percentile(values, 95)
            
            # Contextual assessment based on data distribution
            if current_value >= p95:
                insights.append("🚨 Outlier High (top 5%)")
            elif current_value >= p75:
                insights.append("⚠️ Above Average (upper quartile)")
            elif current_value >= p25:
                insights.append("✅ Normal Range")
            else:
                insights.append("📉 Below Average (lower quartile)")
        
        # Phase 1: Trend detection using linear regression
        if len(values) >= 5:
            slope = np.polyfit(np.arange(len(values)), values, 1)[0]
            if abs(slope) > 0.01:
                if slope > 0:
                    insights.append("📈 Increasing trend")
                else:
                    insights.append("📉 Decreasing trend")
            else:
                insights.append("➡️ Stable trend")
        
        return " | ".join(insights) if insights else ""
        
    except Exception:
        return ""


def _detect_simple_correlations(metric_dfs: Dict[str, Any]) -> List[str]:
    """Detect simple correlations between key metrics - Phase 2.2 enhancement"""
    
    correlations = []
    
    try:
        # Get key metrics for correlation analysis
        gpu_temp = metric_dfs.get("GPU Temperature (°C)")
        gpu_usage = metric_dfs.get("GPU Usage (%)")
        p95_latency = metric_dfs.get("P95 Latency (s)")
        requests_running = metric_dfs.get("Requests Running")
        inference_time = metric_dfs.get("Inference Time (s)")
        
        # Correlation 1: GPU Temperature vs GPU Usage
        if (gpu_temp is not None and not gpu_temp.empty and 
            gpu_usage is not None and not gpu_usage.empty and
            len(gpu_temp) > 3 and len(gpu_usage) > 3):
            
            temp_latest = gpu_temp['value'].iloc[-1]
            usage_latest = gpu_usage['value'].iloc[-1]
            
            # Simple correlation logic
            if temp_latest > 70 and usage_latest > 80:
                correlations.append(f"🔥 High GPU temperature ({temp_latest:.1f}°C) correlates with high utilization ({usage_latest:.1f}%)")
            elif temp_latest > 60 and usage_latest < 20:
                correlations.append(f"🌡️ GPU temperature elevated ({temp_latest:.1f}°C) despite low utilization ({usage_latest:.1f}%) - check cooling")
        
        # Correlation 2: Latency vs Request Queue
        if (p95_latency is not None and not p95_latency.empty and 
            requests_running is not None and not requests_running.empty and
            len(p95_latency) > 3 and len(requests_running) > 3):
            
            latency_latest = p95_latency['value'].iloc[-1]
            queue_latest = requests_running['value'].iloc[-1]
            
            if latency_latest > 3.0 and queue_latest > 10:
                correlations.append(f"⏱️ High latency ({latency_latest:.2f}s) correlates with request queue ({queue_latest:.0f} requests)")
            elif latency_latest > 5.0 and queue_latest < 5:
                correlations.append(f"🐌 High latency ({latency_latest:.2f}s) despite low queue ({queue_latest:.0f}) - model performance issue")
        
        # Correlation 3: Inference Time vs Latency Consistency
        if (inference_time is not None and not inference_time.empty and 
            p95_latency is not None and not p95_latency.empty and
            len(inference_time) > 3 and len(p95_latency) > 3):
            
            inference_latest = inference_time['value'].iloc[-1]
            latency_latest = p95_latency['value'].iloc[-1]
            
            # If P95 latency is much higher than inference time, indicates queuing
            if inference_latest > 0 and latency_latest > (inference_latest * 2):
                correlations.append(f"⏳ P95 latency ({latency_latest:.2f}s) exceeds inference time ({inference_latest:.2f}s) - queuing delays detected")
        
        return correlations
        
    except Exception:
        return []


def _calculate_performance_score(metric_dfs: Dict[str, Any]) -> str:
    """Calculate weighted performance score using ML algorithm - Phase 3.1"""
    
    try:
        # Define MLOps-focused metric weights (sum = 1.0)
        metric_weights = {
            "P95 Latency (s)": 0.35,        # Highest - user experience
            "Inference Time (s)": 0.25,     # Model performance
            "GPU Temperature (°C)": 0.15,   # Hardware health
            "GPU Usage (%)": 0.15,          # Resource efficiency
            "Requests Running": 0.10         # System load
        }
        
        total_score = 0
        total_weight = 0
        component_scores = {}
        
        for metric_name, weight in metric_weights.items():
            if metric_name in metric_dfs:
                df = metric_dfs[metric_name]
                if df is not None and not df.empty:
                    values = df['value'].values
                    current_value = values[-1]
                    
                    # Calculate metric score using percentile-based scoring
                    if len(values) >= 5:
                        p25 = np.percentile(values, 25)
                        p75 = np.percentile(values, 75)
                        p95 = np.percentile(values, 95)
                        
                        # Score based on performance (lower is better for latency/time metrics)
                        if metric_name in ["P95 Latency (s)", "Inference Time (s)"]:
                            if current_value <= p25:
                                score = 100  # Best performance (low latency)
                            elif current_value <= p75:
                                score = 70   # Good performance
                            elif current_value <= p95:
                                score = 40   # Poor performance
                            else:
                                score = 20   # Critical performance
                        else:
                            # For other metrics, higher can be better (GPU usage, etc.)
                            if current_value >= p75:
                                score = 80   # Good utilization
                            elif current_value >= p25:
                                score = 60   # Moderate
                            else:
                                score = 40   # Low utilization
                        
                        component_scores[metric_name] = score
                        total_score += score * weight
                        total_weight += weight
        
        if total_weight == 0:
            return ""
        
        # Calculate final weighted score
        final_score = int(total_score / total_weight)
        
        # Determine status
        if final_score >= 80:
            status = "🟢 EXCELLENT"
        elif final_score >= 60:
            status = "✅ GOOD"
        elif final_score >= 40:
            status = "⚠️ WARNING"
        else:
            status = "🚨 CRITICAL"
        
        # Find bottleneck (lowest scoring component)
        bottleneck = min(component_scores.items(), key=lambda x: x[1]) if component_scores else None
        
        result = f"📊 PERFORMANCE SCORE: {final_score}/100 ({status})"
        if bottleneck:
            result += f" | Bottleneck: {bottleneck[0]} ({bottleneck[1]}/100)"
        
        return result
        
    except Exception:
        return ""


def _predict_threshold_crossings(metric_dfs: Dict[str, Any]) -> List[str]:
    """Predict when metrics will cross critical thresholds - Phase 3.2 ML algorithm"""
    
    predictions = []
    
    # Define critical thresholds for key metrics
    thresholds = {
        "P95 Latency (s)": 10.0,      # 10s is critical for user experience
        "Inference Time (s)": 5.0,    # 5s is too slow for real-time
        "GPU Temperature (°C)": 85.0, # 85°C thermal throttling risk
        "Requests Running": 100.0      # 100 requests = queue saturation
    }
    
    try:
        for metric_name, threshold in thresholds.items():
            if metric_name in metric_dfs:
                df = metric_dfs[metric_name]
                if df is not None and not df.empty and len(df) >= 5:
                    values = df['value'].values
                    current_value = values[-1]
                    
                    # Use linear regression to predict future values
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    
                    # Calculate R-squared for prediction confidence
                    y_pred = slope * x + intercept
                    ss_res = np.sum((values - y_pred) ** 2)
                    ss_tot = np.sum((values - np.mean(values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Only predict if trend is statistically significant
                    if abs(slope) > 0.01 and r_squared > 0.5 and slope > 0:
                        # Calculate time to reach threshold (assuming 5-minute intervals)
                        if current_value < threshold:
                            intervals_to_threshold = (threshold - current_value) / slope
                            minutes_to_threshold = intervals_to_threshold * 5
                            
                            if 0 < minutes_to_threshold <= 120:  # Within 2 hours
                                confidence = "High" if r_squared > 0.8 else "Moderate"
                                predictions.append(f"🚨 PREDICTION: {metric_name} will reach {threshold} in ~{int(minutes_to_threshold)} minutes ({confidence.lower()} confidence, R²={r_squared:.3f})")
                            elif minutes_to_threshold <= 360:  # Within 6 hours
                                hours = minutes_to_threshold / 60
                                predictions.append(f"⚠️ FORECAST: {metric_name} trending toward {threshold} in ~{hours:.1f} hours (R²={r_squared:.3f})")
        
        return predictions
        
    except Exception:
        return []


def summarize_with_llm(
    prompt: str,
    summarize_model_id: str,
    response_type: ResponseType,
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_validation: bool = True,
) -> str:
    """
    Summarize content using an LLM (local or external).
    
    Args:
        prompt: The content to summarize
        summarize_model_id: Model identifier from MODEL_CONFIG
        response_type: Expected response type for validation (OPENSHIFT, VLLM, CHAT) - required
        api_key: API key for external models (optional for local models)
        messages: Previous conversation messages (optional)
        max_tokens: Maximum number of tokens to generate (default: 6000)
        enable_validation: Whether to enable response validation and cleanup (default: True)

    Returns:
        LLM-generated summary text (cleaned if validation enabled)
    """
    headers = {"Content-Type": "application/json"}
    # Get model configuration
    model_info = MODEL_CONFIG.get(summarize_model_id, {})
    is_external = model_info.get("external", False)

    # Building LLM messages array
    llm_messages = []
    if messages:
        llm_messages.extend(messages)
    # Ensure the new prompt is always added as the last user message
    llm_messages.append({"role": "user", "content": prompt})

    if is_external:
        # External model (like OpenAI, Anthropic, etc.)
        if not api_key:
            raise ValueError(
                f"API key required for external model {summarize_model_id}"
            )

        # Get provider-specific configuration
        provider = model_info.get("provider", "openai")
        api_url = model_info.get("apiUrl", "https://api.openai.com/v1/chat/completions")
        model_name = model_info.get("modelName")

        # Provider-specific authentication and payload
        if provider == "google":
            # Google Gemini API format
            headers["x-goog-api-key"] = api_key

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
            }
        else:
            # OpenAI and compatible APIs
            headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": model_name,
                "messages": llm_messages,
                "temperature": DETERMINISTIC_TEMPERATURE,  # Deterministic output
                "max_tokens": max_tokens,
            }

        response_json = _make_api_request(api_url, headers, payload, verify_ssl=DEFAULT_SSL_VERIFICATION)
        raw_response = _validate_and_extract_response(
            response_json, is_external=True, provider=provider
        )

        # For external models, no need to do response validation and cleanup
        return raw_response

    else:
        # Local model (deployed in cluster)
        if LLM_API_TOKEN:
            headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

        # Determine correct local model identifier: prefer serviceName if present
        # summarize_model_id may be a human/registry id (e.g., "meta-llama/..."), while
        # LlamaStack typically expects the backend service name (e.g., "llama-3-2-3b-instruct").
        model_id_to_use = (
            model_info.get("serviceName")
            or model_info.get("modelName")
            or summarize_model_id
        )

        # Combine all messages into a single prompt
        prompt_text = ""
        if messages:
            for msg in messages:
                prompt_text += f"{msg['role']}: {msg['content']}\n"
        prompt_text += prompt  # Add the current prompt
        # Try multiple possible model identifiers to maximize compatibility
        # LlamaStack may expect different model IDs than MODEL_CONFIG keys
        # Priority: serviceName (LlamaStack backend) -> modelName (alt ID) -> summarize_model_id (user key)
        candidate_ids = []
        for candidate in [model_info.get("serviceName"), model_info.get("modelName"), summarize_model_id]:
            if candidate and candidate not in candidate_ids:
                candidate_ids.append(candidate)

        last_err: Optional[Exception] = None
        response_json = None
        # Attempt each candidate model ID until one succeeds
        for candidate_model_id in candidate_ids:
            payload = {
                "model": candidate_model_id,
                "prompt": prompt_text,
                "temperature": DETERMINISTIC_TEMPERATURE,  # Deterministic output
                "max_tokens": max_tokens,
            }
            try:
                response_json = _make_api_request(
                    f"{LLAMA_STACK_URL}/completions", headers, payload, verify_ssl=VERIFY_SSL
                )
                break  # Success - stop trying other candidates
            except requests.exceptions.HTTPError as http_err:  # type: ignore[name-defined]
                # Parse error details to determine if we should try next candidate
                try:
                    status = http_err.response.status_code if http_err.response is not None else None
                    body = http_err.response.text if http_err.response is not None else ""
                except Exception:
                    status, body = None, ""
                
                # Only retry for "model not found" errors; re-raise other HTTP errors immediately
                if status in (400, 404) and ("Model" in body and "not found" in body):
                    last_err = http_err
                    continue  # Try next candidate
                else:
                    raise  # Non-model-related error, fail fast
            except Exception as e:
                # Network/connection errors - save and try next candidate
                last_err = e
                continue

        if response_json is None:
            # All model ID candidates failed
            if last_err:
                raise last_err
            raise RuntimeError("Failed to obtain response from LlamaStack completions endpoint")

        raw_response = _validate_and_extract_response(
            response_json, is_external=False, provider="LLM"
        )

        # Apply response validation and cleanup if enabled
        if enable_validation:
            
            validation_result = ResponseValidator.clean_response(raw_response, response_type, prompt)
            
            return validation_result['cleaned_response']
        else:
            return raw_response





def build_chat_prompt(user_question: str, metrics_summary: str) -> str:
    """Build a chat prompt combining user question with metrics context"""
    prompt = f"""
Metrics Summary:
{metrics_summary}

Question: {user_question}

Provide a concise answer that:
1. Directly answers the question using the metrics data
2. States the current status (Normal/Warning/Critical)
3. Gives brief recommendations if needed

Keep response under 100 words.
Stop after you have provided the answer and do not add additional explanations or notes.

ANSWER:
"""
    return prompt.strip()


def build_prompt(metric_dfs, model_name: str) -> str:
    """Build analysis prompt for vLLM metrics data"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    prompt = f"""
You are a machine learning model performance analysis expert. Please analyze the following vLLM metrics for model '{model_name}' and provide a comprehensive summary.

Current Analysis Time: {current_time}

METRICS DATA:
"""
    
    # Filter to only MLOps-critical metrics using configuration
    priority_metrics = _get_priority_metrics(metric_dfs, max_metrics=8)
    
    for metric_name, df in priority_metrics.items():
        if df is not None and not df.empty:
            prompt += f"\n=== {metric_name.upper()} ===\n"
            # Add DataFrame summary
            prompt += f"Data points: {len(df)}\n"
            if 'value' in df.columns:
                latest_value = df['value'].iloc[-1] if len(df) > 0 else 0
                
                prompt += f"Latest value: {latest_value}\n"
                prompt += f"Average: {df['value'].mean():.2f}\n"
                prompt += f"Min: {df['value'].min():.2f}, Max: {df['value'].max():.2f}\n"
                
                # Add Phase 1 simple intelligence
                intelligence = _add_simple_intelligence(metric_name, df)
                if intelligence:
                    prompt += f"Intelligence: {intelligence}\n"
    
    # Add Phase 2.2: Simple correlation detection
    correlations = _detect_simple_correlations(metric_dfs)
    if correlations:
        prompt += f"\nCORRELATION INSIGHTS:\n"
        for correlation in correlations:
            prompt += f"- {correlation}\n"
    
    # Add Phase 3.1: Performance scoring algorithm
    performance_score = _calculate_performance_score(metric_dfs)
    if performance_score:
        prompt += f"\nPERFORMANCE ANALYSIS:\n- {performance_score}\n"
    
    # Add Phase 3.2: Predictive threshold analysis
    predictions = _predict_threshold_crossings(metric_dfs)
    if predictions:
        prompt += f"\nPREDICTIVE ANALYSIS:\n"
        for prediction in predictions:
            prompt += f"- {prediction}\n"
    
    prompt += """

ANALYSIS REQUIREMENTS:
1. **Performance Summary**: Overall performance score and status,Primary bottleneck identification, Brief health assessment

2. **Key Metrics Analysis**: 
   - List 4-5 most critical metrics with current values
   - Include volatility percentages for each metric
   - Note trend direction (increasing/decreasing/stable)
   - Identify outliers and position status (upper quartile, outlier high, etc.)

3. **Critical Findings**:
   - Most serious statistical findings with specific values
   - Correlation insights (if metrics are related)
   - Outlier detection results
   - Prediction warnings (if threshold crossings expected)

4. **Trends and Patterns**:
   - Statistical trend analysis with R-squared confidence where available
   - Pattern recognition across multiple metrics
   - Volatility assessment and stability analysis

5. **Recommendations**:
   - Specific actionable steps based on statistical analysis
   - Target thresholds and optimization goals
   - Priority-based action items

6. **Attentions**:
   - Top 3 issues requiring immediate focus
   - Include specific statistical values (volatility %, scores, predictions)
   - Risk assessment with quantified metrics

In your response, do not add or ask additional questions. 
Answer each requirement above concisely as a summary in less than 250 words. 
Stop after you have answered requirement 5 and do not add explainations or notes.
Please provide a clear, structured analysis that would be useful for both technical teams and stakeholders.
"""
    
    return prompt.strip()


def build_openshift_prompt(
    metric_dfs, metric_category, namespace=None, scope_description=None
):
    """
    Build prompt for OpenShift metrics analysis
    
    Note: This function depends on describe_trend() and detect_anomalies() 
    which will be moved to core/metrics.py later.
    """
    if scope_description:
        scope = scope_description
    else:
        scope = f"namespace **{namespace}**" if namespace else "cluster-wide"

    header = f"You are an expert in OpenShift platform monitoring and operations. You are evaluating OpenShift **{metric_category}** metrics for {scope}.\n\n📊 **Metrics**:\n"
    analysis_focus = f"{metric_category.lower()} performance and health"

    lines = []
    for label, df in metric_dfs.items():
        if df.empty:
            lines.append(f"- {label}: No data")
            continue
        avg = df["value"].mean()
        latest = df["value"].iloc[-1] if not df.empty else 0
        # TODO: Import these functions from core.metrics when available
        # trend = describe_trend(df)
        # anomaly = detect_anomalies(df, label)
        trend = "stable"  # Placeholder
        anomaly = "normal"  # Placeholder
        lines.append(
            f"- {label}: Avg={avg:.2f}, Latest={latest:.2f}, Trend={trend}, {anomaly}"
        )

    analysis_questions = f"""🔍 Please analyze:
1. What's the current state of {analysis_focus}?
2. Are there any performance or reliability concerns?
3. What actions should be taken?
4. Any optimization recommendations?

Do not add or ask additional questions. Your response should only include the questions and answers for the above questions.
For each question, state the question in bold font, and then answer each question concisely and directly with maximum of 150 words.
If there is no direct answer to a question, say so and do not speculate or add additional information. 
Stop after you have answered question 4 and do not add explainations or notes.
"""
    return f"""{header}
{chr(10).join(lines)}

{analysis_questions}
""".strip()


def build_openshift_metrics_context(
    metric_dfs, metric_category, namespace=None, scope_description=None
):
    """
    Build metrics-only context for OpenShift chat (no analysis questions).

    Returns a header and bullet list of metric summaries without any
    instruction section. Intended for chat-oriented prompts.
    """
    if scope_description:
        scope = scope_description
    else:
        scope = f"namespace **{namespace}**" if namespace else "cluster-wide"

    header = (
        f"You are an expert in OpenShift platform monitoring and operations. "
        f"You are evaluating OpenShift **{metric_category}** metrics for {scope}.\n\n"
        f"📊 **Metrics**:\n"
    )

    lines = []
    for label, df in metric_dfs.items():
        if df.empty:
            lines.append(f"- {label}: No data")
            continue
        avg = df["value"].mean()
        latest = df["value"].iloc[-1] if not df.empty else 0
        trend = "stable"  # Placeholder until describe_trend is available
        anomaly = "normal"  # Placeholder until detect_anomalies is available
        lines.append(
            f"- {label}: Avg={avg:.2f}, Latest={latest:.2f}, Trend={trend}, {anomaly}"
        )

    return f"""{header}
{chr(10).join(lines)}
""".strip()

def build_openshift_chat_prompt(
    question: str,
    metrics_context: str,
    time_range_info: Optional[Dict[str, Any]] = None,
    chat_scope: str = "namespace_specific",
    target_namespace: str = None,
    alerts_context: str = "",
) -> str:
    """Build specialized prompt for OpenShift/Kubernetes queries"""
    # Build scope context
    if chat_scope == CHAT_SCOPE_FLEET_WIDE:
        scope_context = "You are analyzing **fleet-wide OpenShift/Kubernetes metrics across ALL namespaces**.\n\n"
    elif target_namespace:
        scope_context = f"You are analyzing OpenShift/Kubernetes metrics for namespace: **{target_namespace}**.\n\n"
    else:
        scope_context = "You are analyzing OpenShift/Kubernetes metrics.\n\n"
    
    # Build time range context
    time_context = ""
    time_range_syntax = FALLBACK_RATE_SYNTAX  # default
    if time_range_info:
        time_duration = time_range_info.get("duration_str", "")
        time_range_syntax = time_range_info.get("rate_syntax", FALLBACK_RATE_SYNTAX)
        time_context = f"""**🕐 TIME RANGE CONTEXT:**
The user asked about: **{time_duration}**
Use time range syntax `[{time_range_syntax}]` in PromQL queries where appropriate.

"""

    # Common OpenShift metrics for reference
    common_metrics = """**📊 Comprehensive OpenShift/Kubernetes Metrics:**
- Pods: `sum(kube_pod_status_phase{phase="Running"})`, `sum(kube_pod_status_phase{phase="Failed"})`
- Deployments: `sum(kube_deployment_status_replicas_ready)`, `sum(kube_deployment_spec_replicas)`
- Services: `sum(kube_service_info)`, `sum(kube_endpoint_address_available)`
- Jobs: `sum(kube_job_status_active)`, `sum(kube_job_status_succeeded)`, `sum(kube_job_status_failed)`
- Storage: `sum(kube_persistentvolume_info)`, `sum(kube_persistentvolumeclaim_info)`
- Config: `sum(kube_configmap_info)`, `sum(kube_secret_info)`
- Nodes: `sum(kube_node_info)`, `sum(kube_node_status_condition{condition="Ready"})`
- CPU: `100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)`
- Memory: `100 - (sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes) * 100)`
- Containers: `count(count by (image)(container_spec_image))`, `sum(kube_pod_container_status_running)`
- Workloads: `sum(kube_daemonset_status_number_ready)`, `sum(kube_statefulset_status_replicas_ready)`

"""

    return f"""
You are a Senior Site Reliability Engineer (SRE) analyzing OpenShift/Kubernetes metrics and answering the user's question with precise, actionable insights.

{scope_context}{time_context}{common_metrics}

**Current Metrics Status:**
{metrics_context.strip()}

**Current Alert Status:**
{alerts_context.strip()}

User Question: {question}

Provide a concise technical analysis focusing on operational insights and recommendations using only the metrics and data provided.
Respond with JSON format: {{"promql": "relevant_query_if_applicable", "summary": "your_analysis"}}.

""".strip()


def build_flexible_llm_prompt(
    question: str,
    model_name: str,
    metrics_context: str,
    generated_tokens_sum: Optional[float] = None,
    selected_namespace: str = None,
    alerts_context: str = "",
    time_range_info: Optional[Dict[str, Any]] = None,
    chat_scope: str = "namespace_specific",
) -> str:
    """
    Build flexible LLM prompt for various metric analysis scenarios
    
    Note: This function depends on get_vllm_metrics() and add_namespace_filter()
    which will be moved to core/metrics.py and core/utils.py later.
    """
    # Safely handle generated_tokens_sum formatting
    summary_tokens_generated = ""
    if generated_tokens_sum is not None:
        try:
            # Convert to float if it's a string
            if isinstance(generated_tokens_sum, str):
                tokens_value = float(generated_tokens_sum)
            else:
                tokens_value = float(generated_tokens_sum)
            summary_tokens_generated = f"A total of {tokens_value:.2f} tokens were generated across all models and namespaces."
        except (ValueError, TypeError):
            summary_tokens_generated = f"Token generation data: {generated_tokens_sum}"

    # Build scope context
    if chat_scope == CHAT_SCOPE_FLEET_WIDE:
        namespace_context = f"You are analyzing **fleet-wide metrics across ALL namespaces** for model **{model_name}**.\n\n"
    elif selected_namespace:
        namespace_context = f"You are currently focused on the namespace: **{selected_namespace}** and model **{model_name}**.\n\n"
    else:
        namespace_context = ""
    
    # Build time range context for the LLM
    time_context = ""
    time_range_syntax = FALLBACK_RATE_SYNTAX  # default
    if time_range_info:
        time_duration = time_range_info.get("duration_str", "")
        time_range_syntax = time_range_info.get("rate_syntax", FALLBACK_RATE_SYNTAX)
        time_context = f"""**🕐 CRITICAL TIME RANGE REQUIREMENTS:**
The user asked about: **{time_duration}**

**MANDATORY PromQL Syntax Rules:**
✅ ALWAYS add time range `[{time_range_syntax}]` to metrics that need it
✅ For P95/P99 latency: `histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[{time_range_syntax}])) by (le))`  
✅ For rates: `rate(vllm:request_prompt_tokens_created[{time_range_syntax}])`
✅ For averages over time: `avg_over_time(vllm:num_requests_running[{time_range_syntax}])`
❌ NEVER use: `vllm:metric_name{{namespace="...", }}` (trailing comma)
❌ NEVER use: `vllm:metric_name{{namespace="..."}}` (missing time range)

"""

    # TODO: Import get_vllm_metrics() and add_namespace_filter() from core modules when available
    # For now, use placeholder metrics list
    metrics_list = "- Placeholder metrics list (to be replaced with actual metrics from core.metrics)"

    # The task is to analyze and connect the dots.
    return f"""
You are a world-class Senior Production Engineer, an expert in observability and root cause analysis. Your primary skill is correlating different types of telemetry data (metrics, alerts, logs, traces) to form a complete picture of system health and answer user questions with deep, actionable insights.

{namespace_context}{time_context}**Complete Observability Context:**
# Available Metrics:
# {metrics_list}

# Current Metric Status:
{metrics_context.strip()}

# Current Alert Status:
# {alerts_context.strip()}

{summary_tokens_generated.strip()}

{{
  "promqls": ["ALERTS"],
  "summary": "Answer to: {question}"
}}
""".strip()


def extract_time_range_with_info(
    query: str, start_ts: Optional[int], end_ts: Optional[int]
) -> tuple[int, int, Dict[str, Any]]:
    """
    Enhanced time range extraction that DYNAMICALLY parses any time expression from user's question
    Supports historical queries for months/years
    """
    query_lower = query.lower()
    
    # Priority 1: DYNAMIC parsing using regex patterns for any time expression  
    time_patterns = [
        # Pattern: "past/last X minutes/hours/days/weeks/months/years"
        r"(?:past|last|previous)\s+(\d+(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)",
        # Pattern: "X minutes/hours/days/weeks/months/years ago"  
        r"(\d+(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\s+ago",
        # Pattern: "in the past X minutes/hours/days/months/years"
        r"in\s+the\s+past\s+(\d+(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)",
        # Pattern: "over the last X minutes/hours/days/months/years"
        r"over\s+the\s+last\s+(\d+(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)",
        # Pattern: "since X months/years ago"
        r"since\s+(\d+(?:\.\d+)?)\s+(months?|years?)\s+ago",
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, query_lower)
        if match:
            number = float(match.group(1))
            unit = match.group(2)
            
            logger.debug("Dynamic time found: %s %s", number, unit)
            
            # Convert to hours
            if unit.startswith('min'):
                hours = number / 60
                if number == 1:
                    rate_syntax = "1m"
                    duration_str = "past 1 minute"
                elif number < 60:
                    rate_syntax = f"{int(number)}m"
                    duration_str = f"past {int(number)} minutes"
                else:
                    rate_syntax = f"{int(number)}m"
                    duration_str = f"past {number} minutes"
            elif unit.startswith('hour') or unit.startswith('hr'):
                hours = number
                if number == 1:
                    rate_syntax = "1h"
                    duration_str = "past 1 hour"
                else:
                    rate_syntax = f"{int(number)}h" if number == int(number) else f"{number}h"
                    duration_str = f"past {int(number) if number == int(number) else number} hours"
            elif unit.startswith('day'):
                hours = number * 24
                if number == 1:
                    rate_syntax = "1d"
                    duration_str = "past 1 day"
                else:
                    rate_syntax = f"{int(number)}d" if number == int(number) else f"{number}d"
                    duration_str = f"past {int(number) if number == int(number) else number} days"
            elif unit.startswith('week'):
                hours = number * 24 * 7
                if number == 1:
                    rate_syntax = "7d"
                    duration_str = "past 1 week"
                else:
                    days = int(number * 7)
                    rate_syntax = f"{days}d"
                    duration_str = f"past {int(number) if number == int(number) else number} weeks"
            elif unit.startswith('month'):
                hours = number * 24 * 30  # Approximate
                days = int(number * 30)
                rate_syntax = f"{days}d"
                duration_str = f"past {int(number) if number == int(number) else number} months"
            elif unit.startswith('year'):
                hours = number * 24 * 365  # Approximate
                days = int(number * 365)
                rate_syntax = f"{days}d"
                duration_str = f"past {int(number) if number == int(number) else number} years"
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            time_range_info = {
                "duration_str": duration_str,
                "rate_syntax": rate_syntax,
                "hours": hours
            }
            
            logger.debug("Parsed duration: %s -> %s", duration_str, rate_syntax)
            return int(start_time.timestamp()), int(end_time.timestamp()), time_range_info
    
    # Priority 2: Handle special keywords and month names
    special_cases = {
        "yesterday": (24, "1d", "yesterday"),
        "today": (24, "1d", "today"), 
        "last hour": (1, "1h", "past 1 hour"),
        "past hour": (1, "1h", "past 1 hour"),
        "last day": (24, "1d", "past 1 day"),
        "last week": (168, "7d", "past 1 week"),
        "past week": (168, "7d", "past 1 week"),
        "last month": (720, "30d", "past 1 month"),
        "past month": (720, "30d", "past 1 month"),
        "last year": (8760, "365d", "past 1 year"),
        "past year": (8760, "365d", "past 1 year"),
    }
    
    # Handle specific month names (for historical queries)
    current_date = datetime.now()
    month_mapping = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    # Check for month names in query
    for month_name, month_num in month_mapping.items():
        if month_name in query_lower:
            # Calculate time range for the entire month
            current_year = current_date.year
            target_year = current_year
            
            # If the month is in the future this year, assume previous year
            if month_num > current_date.month:
                target_year = current_year - 1
            
            # Get start and end of target month
            if month_num == 12:
                next_month = 1
                next_year = target_year + 1
            else:
                next_month = month_num + 1
                next_year = target_year
                
            month_start = datetime(target_year, month_num, 1)
            month_end = datetime(next_year, next_month, 1) - timedelta(seconds=1)
            
            # Calculate how long ago this was
            time_diff = current_date - month_end
            hours_ago = time_diff.total_seconds() / 3600
            
            time_range_info = {
                "duration_str": f"{month_name.title()} {target_year}",
                "rate_syntax": "1h",  # Use hourly resolution for month-long queries
                "hours": hours_ago,
                "is_historical_month": True
            }
            
            logger.info("Historical month query: %s %s", month_name.title(), target_year)
            return int(month_start.timestamp()), int(month_end.timestamp()), time_range_info
    
    for keyword, (hours, rate_syntax, duration_str) in special_cases.items():
        if keyword in query_lower:
            logger.debug("Special case: %s -> %s hours", keyword, hours)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            time_range_info = {
                "duration_str": duration_str,
                "rate_syntax": rate_syntax,
                "hours": hours
            }
            
            return int(start_time.timestamp()), int(end_time.timestamp()), time_range_info

    # Priority 2: Parse specific dates using dateparser
    found_dates = search_dates(query, settings={"PREFER_DATES_FROM": "past"})

    if found_dates:
        logger.debug("Specific date found in query; building full day range")

        # Take the date part from the first result given by dateparser
        target_date = found_dates[0][1].date()

        # Create "naive" datetime objects for start and end of day
        start_time_naive = datetime.combine(target_date, time.min)
        end_time_naive = datetime.combine(target_date, time.max)

        # Make the datetime objects UTC-aware ---
        start_time_utc = start_time_naive.replace(tzinfo=timezone.utc)
        end_time_utc = end_time_naive.replace(tzinfo=timezone.utc)

        time_range_info = {
                "duration_str": f"on {target_date.strftime('%Y-%m-%d')}",
                "rate_syntax": FALLBACK_RATE_SYNTAX,
                "hours": 24
            }

        return int(start_time_utc.timestamp()), int(end_time_utc.timestamp()), time_range_info

    # Priority 3: Use timestamps from the request if explicitly provided
    if start_ts and end_ts:
        logger.debug("No time in query; using provided timestamps as fallback")
        time_range_hours = (end_ts - start_ts) / 3600
        
        # Use exact time range from timestamps
        if time_range_hours <= 1:
            duration_str = "past 1 hour"
            rate_syntax = "1h"
        elif time_range_hours < 24:
            duration_str = f"past {int(time_range_hours)} hours"
            rate_syntax = f"{int(time_range_hours)}h"
        elif time_range_hours <= 24:
            duration_str = "past 1 day"
            rate_syntax = "1d"
        elif time_range_hours < 168:
            days = int(time_range_hours / 24)
            duration_str = f"past {days} days"
            rate_syntax = f"{days}d"
        else:
            days = int(time_range_hours / 24)
            duration_str = f"past {days} days"
            rate_syntax = f"{days}d"
        
        time_range_info = {
            "duration_str": duration_str,
            "rate_syntax": rate_syntax,
            "hours": time_range_hours
        }
        
        return start_ts, end_ts, time_range_info

    # Priority 4: Fallback to a default time range (last 1 hour)
    logger.debug("No time in query or request; defaulting to last 1 hour")
    now = datetime.now()
    end_time = now
    start_time = end_time - timedelta(hours=DEFAULT_TIME_RANGE_HOURS)
    
    time_range_info = {
        "duration_str": "past 1 hour",
        "rate_syntax": DEFAULT_RATE_SYNTAX,  # Use default rate syntax for fallback
        "hours": DEFAULT_TIME_RANGE_HOURS
    }
    
    return int(start_time.timestamp()), int(end_time.timestamp()), time_range_info


def extract_time_range(
    query: str, start_ts: Optional[int], end_ts: Optional[int]
) -> (int, int):
    """
    Backward compatibility wrapper for extract_time_range_with_info
    """
    start_ts, end_ts, _ = extract_time_range_with_info(query, start_ts, end_ts)
    return start_ts, end_ts


def add_namespace_filter(promql: str, namespace: str) -> str:
    """
    Adds or enforces a `namespace="..."` filter in the PromQL query.
    """
    if f'namespace="{namespace}"' in promql:
        return promql  # Already included

    # If there's a label filter (e.g., `{job="vllm"}`), insert namespace
    if "{" in promql:
        return promql.replace("{", f'{{namespace="{namespace}", ', 1)
    else:
        # No label filter at all, add one
        return f'{promql}{{namespace="{namespace}"}}'


def fix_promql_syntax(promql: str, time_range_syntax: str = FALLBACK_RATE_SYNTAX) -> str:
    """
    Post-process PromQL to fix common syntax issues and ensure proper time range syntax
    """
    if not promql:
        return promql
    
    # Fix trailing commas in label selectors
    promql = re.sub(r',\s*}', '}', promql)
    promql = re.sub(r'{\s*,', '{', promql)
    
    # Fix double commas
    promql = re.sub(r',,+', ',', promql)
    
    # Fix incomplete time range brackets (like [15m without closing bracket)
    promql = re.sub(r'\[(\d+[smhd])\s*$', r'[\1]', promql)
    
    # Ensure proper time range syntax for specific metric types
    if 'latency' in promql.lower() and 'histogram_quantile' not in promql:
        # For latency metrics that should use histogram_quantile
        if 'vllm:e2e_request_latency_seconds_bucket' not in promql:
            if 'vllm:e2e_request_latency_seconds' in promql:
                promql = promql.replace(
                    'vllm:e2e_request_latency_seconds_sum',
                    f'histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[{time_range_syntax}])) by (le))'
                )
    
    # Add time range syntax to rate functions if missing
    if 'rate(' in promql and '[' not in promql:
        promql = re.sub(r'rate\(([^)]+)\)', f'rate(\\1[{time_range_syntax}])', promql)
    
    # For metrics that have time ranges but aren't in rate() functions, convert to rate()
    if '[' in promql and 'rate(' not in promql and 'histogram_quantile' not in promql:
        # Extract the metric and its labels
        pattern = r'([a-zA-Z_:][a-zA-Z0-9_:]*(?:{[^}]*})?)\[([^]]+)\]'
        match = re.search(pattern, promql)
        if match:
            metric_with_labels = match.group(1)
            time_range = match.group(2)
            # Convert to rate() function
            promql = re.sub(pattern, f'rate({metric_with_labels}[{time_range}])', promql)
    
    # Fix namespace label formatting issues
    promql = re.sub(r"namespace='([^']*)'", r'namespace="\1"', promql)
    
    # Ensure proper closing of metric queries
    if promql.endswith('[') or promql.endswith('{'):
        promql = promql.rstrip('[{')
    
    # Balance parentheses - count and add missing closing parentheses
    open_parens = promql.count('(')
    close_parens = promql.count(')')
    if open_parens > close_parens:
        promql += ')' * (open_parens - close_parens)
    
    return promql


def format_alerts_for_ui(
    promql_query: str,
    alerts_data: list,
    alert_definitions: dict = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> str:
    """
    Takes a list of alerts and formats them into a clean, structured
    markdown string suitable for the UI, including alert meanings if available.
    """
    # Format time range if available
    time_range_str = ""
    if start_ts and end_ts:
        try:
            start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M")
            end_str = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M")
            time_range_str = f" between `{start_str}` and `{end_str}`"
        except Exception:
            time_range_str = ""

    summary_lines = [f"PromQL Query for Alerts: `{promql_query}`\n"]
    if not alerts_data:
        summary_lines.append(
            f"No relevant alerts were firing in the specified time range{time_range_str}."
        )
        return "\n".join(summary_lines)

    # Group alerts by name and count instances
    from collections import defaultdict
    alert_groups = defaultdict(lambda: {"count": 0, "severity": "unknown", "example": None})
    
    for alert in alerts_data:
        alert_name = alert.get("alertname", "UnknownAlert")
        alert_groups[alert_name]["count"] += 1
        
        # Keep the highest severity and first example
        if alert_groups[alert_name]["example"] is None:
            alert_groups[alert_name]["example"] = alert
            alert_groups[alert_name]["severity"] = alert.get("severity", "unknown")
        elif alert.get("severity") in ["critical", "warning"] and alert_groups[alert_name]["severity"] not in ["critical", "warning"]:
            alert_groups[alert_name]["severity"] = alert.get("severity", "unknown")

    # Sort by severity (critical first), then by count (highest first)
    def severity_priority(sev):
        return {"critical": 0, "warning": 1, "info": 2}.get(sev, 3)
    
    sorted_alert_items = sorted(
        alert_groups.items(), 
        key=lambda x: (severity_priority(x[1]["severity"]), -x[1]["count"])
    )

    # Limit to top 15 alert types to avoid overwhelming the LLM
    limited_alerts = sorted_alert_items[:15]
    
    summary_lines.append(f"Found {len(alerts_data)} total alerts of {len(alert_groups)} different types. Showing top {len(limited_alerts)} alert types:")

    # Create a concise summary
    for alert_name, alert_info in limited_alerts:
        count = alert_info["count"]
        severity = alert_info["severity"]
        example_alert = alert_info["example"]
        
        namespace = example_alert["labels"].get("namespace", "")
        timestamp = example_alert.get("timestamp", "")
        
        count_text = f"{count} instance{'s' if count > 1 else ''}"
        summary_lines.append(
            f"- **{alert_name}** ({count_text}): Severity **{severity}**"
            + (f", Example from: `{namespace}`" if namespace else "")
            + (f" at `{timestamp}`" if timestamp else "")
        )
    
    if len(alert_groups) > 15:
        summary_lines.append(f"... and {len(alert_groups) - 15} more alert types")

    return "\n".join(summary_lines) 