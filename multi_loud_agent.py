import argparse, json, os, textwrap
from typing import List, Dict, Any

import boto3  # used only if you want to fetch custodian from s3 in future; not required now
from google.cloud import recommender_v1
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain import LLMChain, PromptTemplate

# ---------------------------
# Utilities: parse Cloud Custodian outputs
# ---------------------------
def parse_custodian_reports(custodian_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read custodian JSON outputs from a directory structure:
    custodian_dir/
      <policy-name>/
        resources.json
    Returns a dict keyed by policy-name with list of resource dicts.
    """
    findings = {}
    if not os.path.isdir(custodian_dir):
        raise FileNotFoundError(f"Custodian directory not found: {custodian_dir}")
    for entry in os.listdir(custodian_dir):
        sub = os.path.join(custodian_dir, entry)
        if not os.path.isdir(sub):
            continue
        res_file = os.path.join(sub, "resources.json")
        if not os.path.isfile(res_file):
            continue
        try:
            with open(res_file, "r") as fh:
                data = json.load(fh)
            findings[entry] = data
        except Exception as e:
            print(f"Warning: failed to parse {res_file}: {e}")
            findings[entry] = []
    return findings

# ---------------------------
# Utilities: GCP Recommender
# ---------------------------
def list_gcp_recommendations(project_id: str, location: str = "global", recommender_ids: List[str] = None, max_results: int = 50):
    """
    Query GCP Recommender for one or more recommenders.
    recommender_ids is a list like "google.compute.instance.IdleResourceRecommender"
    """
    if recommender_ids is None:
        # default set commonly useful recommenders
        recommender_ids = [
            "google.compute.instance.IdleResourceRecommender",
            "google.compute.disk.IdleResourceRecommender",
            "google.cloud.sql.high-utilization"  # example subtype; adjust as required
        ]

    client = recommender_v1.RecommenderClient()
    all_recs = []
    for rec_id in recommender_ids:
        parent = f"projects/{project_id}/locations/{location}/recommenders/{rec_id}"
        try:
            pager = client.list_recommendations(request={"parent": parent, "page_size": max_results})
            for r in pager:
                # extract important fields safely
                rec = {
                    "name": r.name,
                    "description": r.description,
                    "recommender_subtype": r.recommender_subtype,
                    "primary_impact": getattr(r.primary_impact, "category", None),
                    "impact": str(r.primary_impact) if r.primary_impact else None,
                    "content": {}
                }
                # content has structured info (resource, operations, costProjection)
                try:
                    rec["content"] = json.loads(r.content.to_json())
                except Exception:
                    # fallback: put string representation
                    rec["content"] = str(r.content)
                all_recs.append(rec)
        except Exception as e:
            print(f"Warning: failed to query recommender {rec_id} for project {project_id}: {e}")
    return all_recs

# ---------------------------
# Terraform snippet generator (very lightweight, heuristic-based)
# - Converts a recommendation to a suggested Terraform code block as text.
# - For safety: only generates "suggestion" strings that need manual review.
# ---------------------------
def generate_terraform_snippets(aws_findings: Dict[str, List[Dict[str, Any]]], gcp_recs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Return dict of {resource_id: terraform_snippet_text}
    This is heuristic-based: looks for common patterns (stop/terminate EC2 -> suggest terraform aws_instance 'count=0' or taint)
    For GCP disk recommendations we suggest google_compute_disk lifecycle or terraform resource removal snippet.
    NOTE: These are suggestions only — review before applying.
    """
    snippets = {}
    # AWS: parse EC2 entries (custodian often includes InstanceId or similar)
    for policy, items in aws_findings.items():
        for r in items:
            # attempt to identify common resource keys
            rid = r.get("InstanceId") or r.get("DBInstanceIdentifier") or r.get("VolumeId") or r.get("id") or r.get("arn") or str(r)[:40]
            # Simplify: produce a suggestion to move to smaller instance or mark as to-be-removed
            if "InstanceId" in r or ("InstanceId" in r.keys()):
                tf = textwrap.dedent(f"""
                # Suggestion for AWS EC2 {rid} (from policy {policy})
                # Confirm current instance type and performance baseline before applying.
                # Example Terraform snippet (manual review required):
                resource "aws_instance" "suggested_{rid.replace('-', '_').lower()}" {{
                  # Replace with actual AMI / subnet / etc. from your configuration
                  ami           = "ami-REPLACE"
                  instance_type = "t3.small" # suggested smaller instance type; verify suitability
                  # consider adding lifecycle prevent_destroy if you want to preserve resource
                  lifecycle {{
                    prevent_destroy = true
                  }}
                }}
                """).strip()
                snippets[rid] = tf
            elif "VolumeId" in r or "VolumeId" in r.keys():
                tf = textwrap.dedent(f"""
                # Suggestion for AWS EBS Volume {rid} (policy {policy})
                # Consider snapshotting and removing if unused. Manual check required.
                # Terraform: remove or set lifecycle rule to delete on removal.
                resource "aws_ebs_volume" "suggested_{rid.replace('-', '_').lower()}" {{
                  availability_zone = "us-east-1a"
                  size              = 8
                  tags = {{
                    Name = "CHECK_{rid}"
                  }}
                }}
                """).strip()
                snippets[rid] = tf
            else:
                # generic placeholder
                snippets[rid] = f"# Suggestion (policy {policy}): Please review resource: {rid}"
    # GCP: basic suggestions based on recommender content
    for rec in gcp_recs:
        name = rec.get("name", "gcp_rec")
        content = rec.get("content", {})
        # Look for resource name inside content
        resource_name = content.get("resourceName") or content.get("instance") or name
        # Example: for an idle VM, suggest terraform google_compute_instance change
        if "Idle" in rec.get("description", "") or "idle" in rec.get("description", "").lower():
            tf = textwrap.dedent(f"""
            # Suggestion for GCP resource {resource_name}
            # This suggestion is based on Recommender entry: {name}
            # Example Terraform snippet (manual review required):
            resource "google_compute_instance" "suggested_{resource_name.replace('/', '_').replace('.', '_')}" {{
              name = "{resource_name.split('/')[-1]}"
              machine_type = "e2-medium" # suggested smaller type; verify suitability
              # ensure you include boot_disk, network_interface, zone, etc.
            }}
            """).strip()
            snippets[name] = tf
        else:
            snippets[name] = f"# GCP recommendation {name}: {rec.get('description','')}"
    return snippets

# ---------------------------
# LangChain tools wrappers
# ---------------------------
def make_custodian_tool(custodian_dir: str) -> Tool:
    def _run(_: str):
        try:
            findings = parse_custodian_reports(custodian_dir)
            # Summarize counts
            summary = {"policy_counts": {k: len(v) for k, v in findings.items()}, "samples": {}}
            # include small sample for each policy
            for k, v in findings.items():
                summary["samples"][k] = v[:3]
            return json.dumps(summary, indent=2, default=str)
        except Exception as e:
            return f"Error reading custodian reports: {e}"
    return Tool(
        name="Read AWS Custodian Reports",
        func=_run,
        description="Reads Cloud Custodian JSON outputs from a directory and returns a JSON summary (policy -> count and small samples). Use this to list under-utilized AWS resources (listing only)."
    )

def make_gcp_recommender_tool(project_id: str) -> Tool:
    def _run(_: str):
        try:
            recs = list_gcp_recommendations(project_id)
            # limit payload size
            small = []
            for r in recs[:50]:
                small.append({
                    "name": r.get("name"),
                    "description": r.get("description"),
                    "recommender_subtype": r.get("recommender_subtype"),
                    "content": r.get("content") if isinstance(r.get("content"), (str, dict)) else str(r.get("content"))[:500]
                })
            return json.dumps({"count": len(recs), "recommendations": small}, indent=2, default=str)
        except Exception as e:
            return f"Error querying GCP Recommender: {e}"
    return Tool(
        name="List GCP Recommender Items",
        func=_run,
        description="Fetches active recommendations from GCP Recommender for the given project. Listing only, no action."
    )

def make_tf_generator_tool() -> Tool:
    def _run(payload_json: str):
        """
        Expects a JSON payload (string) combining custodian findings and gcp recs.
        Returns a set of Terraform snippet suggestions as a JSON map.
        """
        try:
            payload = json.loads(payload_json)
            aws_findings = payload.get("aws_findings", {})
            gcp_recs = payload.get("gcp_recs", [])
            snippets = generate_terraform_snippets(aws_findings, gcp_recs)
            return json.dumps({"snippets": snippets}, indent=2)
        except Exception as e:
            return f"Error generating terraform suggestions: {e}"
    return Tool(
        name="Generate Terraform Snippets",
        func=_run,
        description="Generate suggested Terraform remediation snippets from combined findings (suggestions only). The input is a JSON string containing 'aws_findings' and 'gcp_recs'."
    )

# ---------------------------
# Main agent orchestration
# ---------------------------
def main(custodian_dir: str, gcp_project: str, out: str = None):
    # Build tools
    aws_tool = make_custodian_tool(custodian_dir)
    gcp_tool = make_gcp_recommender_tool(gcp_project)
    tf_tool = make_tf_generator_tool()

    # LLM — replace model name if you use a different provider
    llm = OpenAI(temperature=0)

    tools = [aws_tool, gcp_tool, tf_tool]

    # We will use a zero-shot REACT style agent via LangChain
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Prompt to agent: fetch AWS & GCP lists, summarize, estimate high-level savings, produce governance report and terraform suggestions.
    query = textwrap.dedent("""
    Please:
    1) Read AWS Cloud Custodian reports (list under-utilized resources).
    2) Query GCP Recommender for the given project (list idle/underutilized recommendations).
    3) Aggregate both results and generate a governance report (short executive summary, prioritized list of top 10 cost-saving opportunities, recommended next steps, required approvals).
    4) Produce Terraform remediation snippet suggestions for the top 5 opportunities (as JSON returned by the 'Generate Terraform Snippets' tool). 
    Only listing and suggestions — do NOT perform any cloud actions.
    Return the final report and a JSON block of suggested Terraform snippets.
    """)

    result = agent.run(query)

    # Save to file if requested
    if out:
        with open(out, "w") as fh:
            fh.write(result)
        print(f"Saved agent output to {out}")
    else:
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custodian-dir", required=True, help="Path to Cloud Custodian output directory (folders with resources.json)")
    parser.add_argument("--gcp-project", required=True, help="GCP project id to query Recommender API")
    parser.add_argument("--out", required=False, help="Output file to write the final report")
    args = parser.parse_args()
    main(args.custodian_dir, args.gcp_project, args.out)
