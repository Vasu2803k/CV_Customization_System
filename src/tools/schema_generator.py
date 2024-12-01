import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import json

client = OpenAI()

META_SCHEMA = {
  "name": "metaschema",
  "schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The name of the schema"
      },
      "type": {
        "type": "string",
        "enum": [
          "object",
          "array",
          "string",
          "number",
          "boolean",
          "null"
        ]
      },
      "properties": {
        "type": "object",
        "additionalProperties": {
          "$ref": "#/$defs/schema_definition"
        }
      },
      "items": {
        "anyOf": [
          {
            "$ref": "#/$defs/schema_definition"
          },
          {
            "type": "array",
            "items": {
              "$ref": "#/$defs/schema_definition"
            }
          }
        ]
      },
      "required": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "additionalProperties": {
        "type": "boolean"
      }
    },
    "required": [
      "type"
    ],
    "additionalProperties": False,
    "if": {
      "properties": {
        "type": {
          "const": "object"
        }
      }
    },
    "then": {
      "required": [
        "properties"
      ]
    },
    "$defs": {
      "schema_definition": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "object",
              "array",
              "string",
              "number",
              "boolean",
              "null"
            ]
          },
          "properties": {
            "type": "object",
            "additionalProperties": {
              "$ref": "#/$defs/schema_definition"
            }
          },
          "items": {
            "anyOf": [
              {
                "$ref": "#/$defs/schema_definition"
              },
              {
                "type": "array",
                "items": {
                  "$ref": "#/$defs/schema_definition"
                }
              }
            ]
          },
          "required": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "additionalProperties": {
            "type": "boolean"
          }
        },
        "required": [
          "type"
        ],
        "additionalProperties": False,
        "if": {
          "properties": {
            "type": {
              "const": "object"
            }
          }
        },
        "then": {
          "required": [
            "properties"
          ]
        }
      }
    }
  }
}

META_PROMPT = """
# Instructions
Return a valid schema for the described JSON.

You must also make sure:
- all fields in an object are set as required
- I REPEAT, ALL FIELDS MUST BE MARKED AS REQUIRED
- all objects must have additionalProperties set to false
    - because of this, some cases like "attributes" or "metadata" properties that would normally allow additional properties should instead have a fixed set of properties
- all objects must have properties defined
- field order matters. any form of "thinking" or "explanation" should come before the conclusion
- $defs must be defined under the schema param

Notable keywords NOT supported include:
- For strings: minLength, maxLength, pattern, format
- For numbers: minimum, maximum, multipleOf
- For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
- For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems

Other notes:
- definitions and recursion are supported
- only if necessary to include references e.g. "$defs", it must be inside the "schema" object

# Examples
Input: Generate a math reasoning schema with steps and a final answer.
Output: {
    "name": "math_reasoning",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "A sequence of steps involved in solving the math problem.",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Description of the reasoning or method used in this step."
                    },
                    "output": {
                        "type": "string",
                        "description": "Result or outcome of this specific step."
                    }
                },
                "required": [
                    "explanation",
                    "output"
                ],
                "additionalProperties": false
            }
        },
        "final_answer": {
            "type": "string",
            "description": "The final solution or answer to the math problem."
        }
    },
    "required": [
        "steps",
        "final_answer"
    ],
    "additionalProperties": false
}

Input: Give me a linked list
Output: {
    "name": "linked_list",
    "type": "object",
    "properties": {
        "linked_list": {
            "$ref": "#/$defs/linked_list_node",
            "description": "The head node of the linked list."
        }
    },
    "$defs": {
        "linked_list_node": {
            "type": "object",
            "description": "Defines a node in a singly linked list.",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The value stored in this node."
                },
                "next": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/linked_list_node"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "description": "Reference to the next node; null if it is the last node."
                }
            },
            "required": [
                "value",
                "next"
            ],
            "additionalProperties": false
        }
    },
    "required": [
        "linked_list"
    ],
    "additionalProperties": false
}

Input: Dynamically generated UI
Output: {
    "name": "ui",
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "The type of the UI component",
            "enum": [
                "div",
                "button",
                "header",
                "section",
                "field",
                "form"
            ]
        },
        "label": {
            "type": "string",
            "description": "The label of the UI component, used for buttons or form fields"
        },
        "children": {
            "type": "array",
            "description": "Nested UI components",
            "items": {
                "$ref": "#"
            }
        },
        "attributes": {
            "type": "array",
            "description": "Arbitrary attributes for the UI component, suitable for any element",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the attribute, for example onClick or className"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value of the attribute"
                    }
                },
                "required": [
                    "name",
                    "value"
                ],
                "additionalProperties": false
            }
        }
    },
    "required": [
        "type",
        "label",
        "children",
        "attributes"
    ],
    "additionalProperties": false
}
""".strip()

def generate_schema(description: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_schema", "json_schema": META_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "Description:\n" + description,
            },
        ],
    )

    return json.loads(completion.choices[0].message.content)

if __name__ == "__main__":
    print(generate_schema("""      You are a Resume Content Optimization Specialist dedicated to transforming resumes into precisely targeted, high-impact documents that maximize candidate potential while maintaining absolute authenticity. Your core responsibility is to transform resume content into powerful, achievement-focused statements written in first person without personal pronouns (I, me, my), creating direct and impactful statements that align precisely with job requirements while drawing exclusively from the candidate's actual background and experience. You must maintain complete authenticity while optimizing resume content, ensuring all content is tailored specifically to match job requirements and skills, never fabricating or embellishing experiences but rather presenting authentic accomplishments in the most compelling and relevant way possible.

      # Primary Directive
      Produce a strategically optimized resume that:
      - Perfectly aligns with target job requirements
      - Highlights candidate's most compelling achievements
      - Ensures maximum ATS compatibility
      - Maintains 100% fidelity to original content
      - Elevates professional narrative

      # Optimization Process

      1. Initial Analysis and Content Enhancement
        - Review current resume structure and content against requirements
        - Document content gaps and opportunities
        - Identify and amplify:
          - Quantifiable achievements
          - Relevant technical skills
          - Industry-specific keywords
          - Demonstrable impact metrics
        - Assess impact statement strength

      2. Strategic Optimization
        - Action Verb Enhancement
          - Replace passive language with dynamic, results-oriented verbs
          - Align verbs with industry and role expectations
          - Ensure verb variety and precision
        
        - Keyword Integration
          - Map and implement job description keywords strategically
          - Achieve 3-5% keyword density
          - Optimize keyword placement for ATS scanning
          - Maintain natural, authentic language flow

        - Impact Amplification
          - Surface and highlight numerical achievements
          - Contextualize metrics (percentages, amounts, scale)
          - Connect metrics to business outcomes
          - Strengthen technical descriptions

      3. ATS Compatibility and Formatting
        - Implement ATS-friendly structure:
          - Standard section headings
          - Clean, consistent formatting
          - Optimal bullet point construction
          - Maximum readability
        - Verify formatting structure
        - Test keyword recognition
        - Validate section headers

      4. Quality Assurance
        - Verify requirements:
          - Minimum 4 substantive bullet points per role
          - Keyword match rate > 80%
          - Clear progression and growth narrative
          - Consistent professional tone
          - Compelling, concise content
        - Run content requirement checks
        - Document improvements and optimization metrics
        - Flag any concerns

      5. Resume Scoring:
        - Score each section based on completeness and quality based on the following criteria:
          * Relevance to target role/industry
          * Action verbs
          * Keywords and industry terminology
          * Formatting consistency
          * Metrics and numbers
          * Effectiveness of achievements and impact statements
          * Progression and growth demonstrated
          * Consistency in narrative and presentation
          * Clarity and readability of content
          * Chronological and logical flow of information
          * Overall coherence and professionalism
        - Generate a summary score for the entire resume

      6. Job Alignment Analysis:
        - Calculate the overall match score considering resume score and job requirements
        - Analyze skill and project gaps and provide recommendations:
          * Identify missing required skills from resume
          * Categorize missing skills by priority (critical/preferred)
          * Estimate learning time for each missing skill
          * Calculate skill acquisition trade-offs based on:
            - Priority level of the skill
            - Time investment needed to learn
            - Impact on job qualification
          * Recommend relevant projects from database:
            - Map projects to missing skills
            - Prioritize projects by skill coverage
            - Estimate project completion timeframes
            - Calculate project implementation trade-offs:
              > Skills gained vs time invested
              > Priority of covered skills
              > Portfolio impact potential
      
      # Notes

      - Ensure zero content fabrication and total authenticity.
      - Focus on strategic amplification and candidate empowerment.
      - Maintain the original meaning while achieving measurable improvements.
"""))