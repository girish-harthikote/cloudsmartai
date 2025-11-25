# cloudsmartai

Multi-cloud LangChain agent:
 - AWS: reads Cloud Custodian JSON "list" outputs from a directory
 - GCP: queries Recommender API for project recommendations
 - Uses an LLM (LangChain/OpenAI) to aggregate results, estimate savings, produce governance report
 - Produces Terraform remediation snippets (suggestions only) for review

Usage:
  export OPENAI_API_KEY=...
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
  python multi_cloud_agent.py \
        --custodian-dir ./custodian_reports \
        --gcp-project my-gcp-project \
        --out report.md
"""
