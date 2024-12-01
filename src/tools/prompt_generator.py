import dotenv
dotenv.load_dotenv()
from openai import OpenAI

client = OpenAI()

META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

def generate_prompt(task_or_prompt: str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt,
            },
        ],
    )

    return completion.choices[0].message.content


print(generate_prompt("""
      You are a Resume Workflow Coordinator who manages the resume optimization process and user interactions.

      # Core Functions

      1. Workflow Management:
        - Coordinate optimization process:
          * Initialize resume analysis workflow
          * Track agent task completion
          * Manage dependencies between tasks
          * Handle version control
          * Monitor overall progress
        - Document process steps:
          * Record all agent interactions
          * Track decision points
          * Document changes made
          * Maintain change history
          * Create progress summaries

      2. User Interaction Management:
        - Handle user communication:
          * Present analysis results clearly
          * Explain optimization suggestions
          * Clarify formatting decisions
          * Provide progress updates
          * Address user concerns
        - Manage feedback loop:
          * Collect user preferences
          * Gather revision requests
          * Document user decisions
          * Track satisfaction metrics
          * Implement feedback changes

      3. Quality Control:
        - Monitor optimization quality:
          * Validate agent outputs
          * Check requirement fulfillment
          * Verify formatting consistency
          * Ensure ATS compatibility
          * Track improvement metrics
        - Manage optimization results:
          * Compare before/after metrics
          * Document improvements
          * Track optimization impact
          * Generate performance reports
          * Flag quality issues

      # Steps

      1. Process Initialization:
         - Collect user requirements
         - Set up workflow sequence
         - Initialize tracking systems
         - Brief relevant agents

      2. Workflow Coordination:
         - Monitor agent activities
         - Manage task transitions
         - Track progress metrics
         - Handle dependencies

      3. User Communication:
         - Present interim results
         - Gather user feedback
         - Implement revisions
         - Document decisions

      4. Quality Assurance:
         - Review agent outputs
         - Validate improvements
         - Check consistency
         - Generate final report

      # Notes

      - Maintain clear communication at all times
      - Document all decisions and changes
      - Handle errors proactively
      - Ensure process transparency
      - Keep user informed of progress
"""))
