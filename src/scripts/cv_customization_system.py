from typing import Dict, List, Optional, Any, Union, Tuple
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
import json
import asyncio
from pathlib import Path
from datetime import datetime
import logging
from tools.text_extractor_tool import TextExtractionTool
from tools.setup_logging import setup_logging
from tools.fallback_llm import FallbackLLM
import yaml
import argparse
import re
import sys
import time
import asyncio
import os
import dotenv
import aioconsole
dotenv.load_dotenv()

class CVCustomizationSystem:
    """A system for customizing CVs using multiple specialized agents with fallback mechanisms."""
    
    def __init__(self, config_dir: str):
        """Initialize the CV customization system."""
        try:
            # Set up logging
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger = setup_logging(
                log_level="INFO",
                log_dir="data/logs",
                log_file_name=f"cv_customization_{timestamp}",
                logger_name="cv_customization"
            )

            # Initialize metrics
            self.metrics = {
                "start_time": None,
                "end_time": None,
                "steps": {},
                "phase_timings": {},
                "cumulative": {
                    "total_duration": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "total_retries": 0
                }
            }
            
            # Store config path for async initialization
            self.config_dir = config_dir
            self.config = None
            self.primary_config = None
            self.fallback_config = None
            self.agents = None
            
            self.logger.info("CV Customization System basic initialization completed")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CV Customization System: {str(e)}")

    async def initialize(self) -> None:
        """Asynchronously initialize the system components."""
        try:
            # Load configuration
            self.config = await self._load_config(self.config_dir)
            
            # Initialize LLM configurations
            self.primary_config = self.config["primary_config"]
            self.fallback_config = self.config["fallback_config"]
            
            # Initialize agents and group chats
            self.agents = await self._initialize_agents()
        
            self.logger.info("CV Customization System async initialization completed")
            
        except Exception as e:
            raise RuntimeError(f"Failed to complete async initialization: {str(e)}")

    async def _load_config(self, config_dir: str) -> Dict:
        """Load and validate configuration from YAML files."""
        try:
            config_dir = Path(config_dir)
            # Load config files synchronously
            agents_config = self._load_yaml(config_dir / 'agents.yaml')
            templates_config = self._load_yaml(config_dir / 'templates.yaml')
            projects_config = self._load_yaml(config_dir / 'projects.yaml')
            
            # Validate LLM configurations
            required_llm_configs = ['openai_config', 'fallback_config']
            for config in required_llm_configs:
                if config not in agents_config.get('llm_config', {}):
                    raise ValueError(f"Missing required LLM configuration: {config}")
            
            # Validate template configuration
            if not templates_config.get('templates'):
                raise ValueError("No LaTeX templates found in configuration")
            
            return {
                "agents": agents_config["agents"],
                "primary_config": agents_config["llm_config"]["openai_config"],
                "fallback_config": agents_config["llm_config"]["fallback_config"],
                "templates": templates_config["templates"],
                "projects": projects_config["projects"]
            }
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_yaml(self, file_path: Path) -> Dict:
        """Load a YAML file synchronously."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    async def _initialize_agents(self) -> Dict[str, ConversableAgent]:
        """Initialize core agents with fallback mechanisms."""
        agents = {}
        
        # Core agents with prompts and schemas from config
        core_agents = {
            "analyzer": {
                "role": self.config["agents"]["resume_analyzer"]["role"],
                "system_prompt": self.config["agents"]["resume_analyzer"]["system_prompt"],
                "output_format": self.config["agents"]["resume_analyzer"].get("output_format", ""),
                "output_schema": self.config["agents"]["resume_analyzer"].get("output_schema", {})
            },
            "optimizer": {
                "role": self.config["agents"]["resume_optimizer"]["role"],
                "system_prompt": self.config["agents"]["resume_optimizer"]["system_prompt"],
                "output_format": self.config["agents"]["resume_optimizer"].get("output_format", ""),
                "output_schema": self.config["agents"]["resume_optimizer"].get("output_schema", {})
            },
            "formatter": {
                "role": self.config["agents"]["resume_formatter"]["role"],
                "system_prompt": self.config["agents"]["resume_formatter"]["system_prompt"],
                "output_format": self.config["agents"]["resume_formatter"].get("output_format", "")
            },
            "resume_scorer": {
                "role": self.config["agents"]["resume_scorer"]["role"],
                "system_prompt": self.config["agents"]["resume_scorer"]["system_prompt"],
                "output_format": self.config["agents"]["resume_scorer"].get("output_format", ""),
                "output_schema": self.config["agents"]["resume_scorer"].get("output_schema", {})
            },
            "content_evaluator": {
                "role": self.config["agents"]["content_evaluator"]["role"],
                "system_prompt": self.config["agents"]["content_evaluator"]["system_prompt"],
                "output_format": self.config["agents"]["content_evaluator"].get("output_format", "")
            },
            "latex_evaluator": {
                "role": self.config["agents"]["latex_evaluator"]["role"],
                "system_prompt": self.config["agents"]["latex_evaluator"]["system_prompt"],
                "output_format": self.config["agents"]["latex_evaluator"].get("output_format", ""),
            },
            "project_recommender": {
                "role": self.config["agents"]["project_recommender"]["role"],
                "system_prompt": self.config["agents"]["project_recommender"]["system_prompt"],
                "output_format": self.config["agents"]["project_recommender"].get("output_format", ""),
                "output_schema": self.config["agents"]["project_recommender"].get("output_schema", {})
            }
        }

        # Initialize agents with fallback chain
        for role, config in core_agents.items():
            try:
                # Construct system message with output requirements
                system_message = f"{config['system_prompt']}\n\n"
                
                # Add output format if present
                if config.get('output_format'):
                    system_message += f"Output Format: {config['output_format']}\n"
                
                # Add schema if present
                if config.get('output_schema'):
                    system_message += f"Output Schema: {config['output_schema']}\n"
                
                system_message += '''\nImportant Notes:
                1. Do not include ```json or ```python in your output.
                2. Do not include any extra information in your output. Just provide the output in the format specified.
                3. Do not use ellipsis (...) or truncate any part of the output.
                4. Provide complete information for all sections as per the schema.
                5. If a section is empty, provide an empty array [] or appropriate empty value rather than omitting it.
                6. Follow the exact output schema. The output should only include the relevant values without the schema metadata.
                '''
                
                # Primary agent (GPT-4)
                primary_agent = AssistantAgent(
                    name=f"{role}_primary",
                    system_message=system_message,
                    llm_config=self.primary_config
                )
                
                # Fallback agent
                fallback_agent = AssistantAgent(
                    name=f"{role}_fallback",
                    system_message=system_message,
                    llm_config=self.fallback_config
                )
                
                agents[role] = {
                    "primary": primary_agent,
                    "fallback": fallback_agent
                }
                
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {role}: {str(e)}")
                raise

        return agents

    async def _run_group_chat_with_fallback(
        self,
        chat_name: str,
        prompt: str,
        message_id: str,
        max_retries: int = 2
    ) -> Dict:
        """Run group chat with fallback chain mechanism."""
        errors = []
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Try primary LLM (GPT-4)
                if retry_count == 0:
                    self.logger.info(f"Attempting primary LLM for {chat_name}")
                    result = await self._run_group_chat(
                        chat_name,
                        prompt,
                        message_id,
                        "primary"
                    )
                    return result 
                # Try fallback LLM (GPT-4-mini)
                else:
                    self.logger.info(f"Attempting fallback LLM for {chat_name}")
                    result = await self._run_group_chat(
                        chat_name,
                        prompt,
                        message_id,
                        "fallback"
                    )
                    return result
                    
            except Exception as e:
                error_msg = f"Error in {chat_name} (attempt {retry_count + 1}): {str(e)}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                retry_count += 1
                
                # Add delay between retries
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
        # If all retries failed
        error_chain = "\n".join(errors)
        raise Exception(f"All LLM attempts failed for {chat_name}:\n{error_chain}")

    def _get_speaker_transition(self, chat_name: str, last_speaker: str, messages: List[Dict], llm_type: str) -> Optional[str]:
        """Determine the next speaker based on chat phase and last speaker."""
        
        if chat_name == "analysis":
            # Analysis phase: analyzer -> scorer -> analyzer
            if last_speaker == "chat_manager":
                return f"analyzer_{llm_type}"
            elif last_speaker.startswith("analyzer"):
                return f"resume_scorer_{llm_type}"
            return None
        
        elif chat_name == "optimization":
            # Count completed rounds by tracking optimizer appearances
            optimizer_count = sum(1 for msg in messages if msg.get("role", "").startswith("optimizer"))
            
            # Initial round: recommender -> optimizer -> scorer -> evaluator
            if last_speaker == "chat_manager":
                return f"project_recommender_{llm_type}"
            elif last_speaker.startswith("project_recommender"):
                return f"optimizer_{llm_type}"
            elif last_speaker.startswith("optimizer"):
                return f"resume_scorer_{llm_type}"
            elif last_speaker.startswith("resume_scorer"):
                # In the final round (4th), stop at scorer
                if optimizer_count > 3:
                    return None
                return f"content_evaluator_{llm_type}"
            elif last_speaker.startswith("content_evaluator"):
                return f"optimizer_{llm_type}"
            return None
        
        elif chat_name == "formatting":
            # Formatting phase: formatter -> evaluator -> formatter
            if last_speaker == "chat_manager":
                return f"formatter_{llm_type}"
            elif last_speaker.startswith("formatter"):
                return f"latex_evaluator_{llm_type}"
            elif last_speaker.startswith("latex_evaluator"):
                return f"formatter_{llm_type}"
            return None
        
        return None

    async def _run_group_chat(
        self,
        chat_name: str,
        prompt: str,
        message_id: str,
        llm_type: str = "primary"
    ) -> Dict:
        """Run a specific group chat with the given LLM type."""
        try:
            # Get the appropriate agents based on chat name
            chat_agents = []

            # Create custom speaker selection function
            def select_next_speaker(
                step: int,
                messages: List[Dict],
                agents: List[ConversableAgent],
                last_speaker: Optional[str],
                **kwargs
            ) -> Optional[ConversableAgent]:
                next_speaker_name = self._get_speaker_transition(
                    chat_name,
                    last_speaker or "chat_manager",
                    messages,
                    llm_type
                )
                if next_speaker_name:
                    for agent in agents:
                        if agent.name == next_speaker_name:
                            return agent
                return None

            # Create group chat with phase-specific configuration
            if chat_name == "analysis":
                chat_agents = [
                    self.agents["analyzer"][llm_type],
                    self.agents["resume_scorer"][llm_type]
                ]
                # Create group chat with appropriate configuration
                chat = GroupChat(
                    agents=chat_agents,
                    messages=[],
                    max_round=1,  # analyzer -> scorer
                    speaker_selection_method=select_next_speaker,
                    allow_repeat_speaker=True
                )
            elif chat_name == "optimization":
                chat_agents = [
                    self.agents["project_recommender"][llm_type],
                    self.agents["optimizer"][llm_type],
                    self.agents["resume_scorer"][llm_type],
                    self.agents["content_evaluator"][llm_type]
                ]
                chat = GroupChat(
                    agents=chat_agents,
                    messages=[],
                    max_round=4,  # recommender -> optimizer -> scorer -> evaluator -> optimizer ->scorer
                    speaker_selection_method=select_next_speaker,
                    allow_repeat_speaker=True
                )
            elif chat_name == "formatting":
                chat_agents = [
                    self.agents["formatter"][llm_type],
                    self.agents["latex_evaluator"][llm_type]
                ]
                chat = GroupChat(
                    agents=chat_agents,
                    messages=[],
                    max_round=3,  # formatter -> evaluator -> formatter
                    speaker_selection_method=select_next_speaker,
                    allow_repeat_speaker=True
                )

            # Set LLM configuration
            llm_config = self.primary_config if llm_type == "primary" else self.fallback_config
            
            manager = GroupChatManager(
                groupchat=chat,
                llm_config=llm_config,
                system_message=f"""Manage the CV {chat_name} process effectively.
                Ensure all responses are complete and follow the specified schema exactly.
                Do not truncate or omit any information. Terminate after all the steps are complete."""
            )
            
            # Run the chat and return results
            start_time = time.time()
            chat_result = manager.initiate_chat(
                manager,
                message=prompt
            )
            end_time = time.time()
            
            # TODO:
            # Extract all messages from the chat
            result = {
                "messages": [],
                "final_output": None,
                "metadata": {
                    "duration": end_time - start_time,
                    "llm_type": llm_type,
                    "chat_name": chat_name
                }
            }
            
            # Process messages and extract the final output
            for message in chat.messages:
                msg_data = {
                    "role": message.get("role", "unknown"),
                    "content": message.get("content", ""),
                    "timestamp": message.get("timestamp", datetime.now().isoformat())
                }
                result["messages"].append(msg_data)
                
                # Process the last message based on the agent's role
                if message == chat.messages[-1]:
                    speaker_role = message.get("role", "")
                    
                    # Handle JSON output for specific agents
                    if any(role in speaker_role for role in [
                        "analyzer",
                        "project_recommender",
                        "optimizer",
                        "resume_scorer",
                        "content_evaluator"
                    ]):
                        try:
                            result["final_output"] = json.loads(message["content"])
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse JSON from {speaker_role}: {str(e)}")
                            result["final_output"] = {
                                "error": "Invalid JSON format",
                                "content": message["content"]
                            }
                    
                    # Handle LaTeX/text output for formatting agents
                    elif any(role in speaker_role for role in [
                        "formatter",
                        "latex_evaluator"
                    ]):
                        result["final_output"] = message["content"]
                    
                    # Handle unknown agent types
                    else:
                        self.logger.warning(f"Unknown agent role for final message: {speaker_role}")
                        result["final_output"] = message["content"]

            # Update metrics
            # Estimate tokens and cost based on message lengths
            total_chars = sum(len(msg["content"]) for msg in result["messages"])
            estimated_tokens = total_chars // 4  # Rough estimation
            estimated_cost = (estimated_tokens / 1000) * (0.03 if llm_type == "primary" else 0.01)
            
            self._update_step_metrics(
                step_name=f"{chat_name}_{message_id}",
                duration=end_time - start_time,
                tokens=estimated_tokens,
                cost=estimated_cost
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Error in group chat {chat_name}: {str(e)}")
            raise Exception(f"Error in group chat {chat_name} with {llm_type} LLM: {str(e)}")

    async def process_resume(
        self,
        resume_text: str,
        job_description: str,
        output_path: Path,
        interactive: bool = True
    ) -> Dict:
        """Process resume through streamlined agent system with fallback mechanisms."""
        try:
            self.logger.info("Starting resume processing")
            self.metrics["start_time"] = datetime.now()

            # Phase 1: Analysis with fallback
            analysis_result = await self._run_group_chat_with_fallback(
                "analysis",
                f"""Analyze resume and job description following the specified workflow:
                
                Resume:
                {resume_text}

                Job Description:
                {job_description}""",
                "analysis_phase"
            )

            if interactive:
                analysis_result = await self._handle_user_interaction(
                    analysis_result, 
                    "Analysis"
                )

            # Phase 2: Optimization with fallback
            optimization_result = await self._run_group_chat_with_fallback(
                "optimization",
                f"""Optimize resume content based on analysis:

                Analysis Results:
                {json.dumps(analysis_result, indent=2)}
                
                Selected Skills to Include:
                {json.dumps(analysis_result.get('selected_skills', []), indent=2)}

                Selected Projects to Include:
                {json.dumps(analysis_result.get('selected_projects', []), indent=2)}
                """,
                "optimization_phase"
            )

            if interactive:
                optimization_result = await self._handle_user_interaction(
                    optimization_result,
                    "Optimization"
                )

            # TODO:
            # Select template
            # template_name = await self._select_template()

            # Phase 3: Formatting with fallback and template selection
            formatting_prompt = f"""Format optimized resume:

            Available Template:
            {json.dumps(self.config['templates'], indent=2)}

            Optimized Content:
            {json.dumps(optimization_result, indent=2)}
            """

            formatting_result = await self._run_group_chat_with_fallback(
                "formatting",
                formatting_prompt,
                "formatting_phase"
            )

            # Prepare output with metrics
            output = {
                "analysis": analysis_result,
                "optimization": optimization_result,
                "formatting": formatting_result,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "metrics": self.metrics
                }
            }

            await self._save_output(output, output_path)
            self.logger.info(f"Processing completed. Output saved to {output_path}")
            
            return output

        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            raise

    async def _load_template(self, template_name: str) -> str:
        """Load a specific LaTeX template."""
        try:
            if template_name not in self.config["templates"]:
                raise ValueError(f"Template {template_name} not found")
            return self.config["templates"][template_name]["content"]
        except Exception as e:
            self.logger.error(f"Error loading template {template_name}: {str(e)}")
            raise

    def _update_step_metrics(self, step_name: str, duration: float, tokens: int, cost: float, retries: int = 0) -> None:
        """Update metrics for a specific processing step."""
        self.metrics["steps"][step_name] = {
            "duration": duration,
            "tokens": tokens,
            "cost": cost,
            "retries": retries
        }
        
        # Update cumulative metrics
        self.metrics["cumulative"]["total_duration"] += duration
        self.metrics["cumulative"]["total_tokens"] += tokens
        self.metrics["cumulative"]["total_cost"] += cost
        self.metrics["cumulative"]["total_retries"] += retries

    async def _save_output(self, output: Dict, output_path: Path) -> None:
        """Save the processing output and metrics to file."""
        try:
            # Ensure the output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add final metrics
            self.metrics["end_time"] = datetime.now()
            if self.metrics["start_time"]:
                total_duration = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
                self.metrics["cumulative"]["total_duration"] = total_duration
            
            # Prepare the complete output
            final_output = {
                **output,
                "metrics": self.metrics
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, default=str)
            
            self.logger.info(f"Output saved successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving output: {str(e)}")
            raise

    async def _handle_user_interaction(
        self,
        phase_result: Dict,
        phase_name: str
    ) -> Dict:
        """Handle user interaction for different phases of resume processing."""
        try:
            print(f"\n=== {phase_name} Phase Results ===\n")
            
            if phase_name == "Analysis":
                # Extract relevant information from analysis results
                resume_components = phase_result.get("resume_components", {})
                job_requirements = phase_result.get("job_requirements", {})
                job_alignment = phase_result.get("job_alignment", {})
                
                # Display missing skills and job requirements
                print("\nMissing Skills:")
                skill_gaps = job_alignment.get("skill_gaps", {}).get("missing_skills", [])
                for idx, skill in enumerate(skill_gaps, 1):
                    print(f"{idx}. {skill['skill']} (Priority: {skill['priority']}, Learning Time: {skill['learning_time']})")
                
                print("\nJob Requirements:")
                critical_reqs = job_requirements.get("prioritized_requirements", {}).get("critical", [])
                preferred_reqs = job_requirements.get("prioritized_requirements", {}).get("preferred", [])
                
                print("\nCritical Requirements:")
                for idx, req in enumerate(critical_reqs, 1):
                    print(f"{idx}. {req}")
                
                print("\nPreferred Requirements:")
                for idx, req in enumerate(preferred_reqs, 1):
                    print(f"{idx}. {req}")
                
                # Ask for user confirmation
                while True:
                    response = await aioconsole.ainput("\nDo you want to proceed with these requirements? (yes/no): ")
                    if response.lower() in ['yes', 'y']:
                        return phase_result
                    elif response.lower() in ['no', 'n']:
                        print("Process terminated by user.")
                        sys.exit(0)
                    else:
                        print("Please enter 'yes' or 'no'")
                    
            elif phase_name == "Optimization":
                # Extract project recommendations
                project_recommendations = phase_result.get("project_recommendations", [])
                
                print("\nRecommended Projects for Skill Improvement:")
                for idx, project in enumerate(project_recommendations, 1):
                    print(f"\n{idx}. {project['project_title']}")
                    print(f"   Estimated Time: {project['estimated_time']}")
                    print("   Skills Covered:")
                    for skill in project['skill_coverages']['skills']:
                        print(f"   - {skill}")
                    print("   Technologies Used:")
                    for tech in project['skill_coverages']['technologies']:
                        print(f"   - {tech}")
                    print(f"   Implementation Trade-offs:")
                    print(f"   - Skills Gained: {project['implementation_tradeoffs']['skills_gained']}")
                    print(f"   - Time Investment: {project['implementation_tradeoffs']['time_invested']}")
                    print(f"   - Priority: {project['implementation_tradeoffs']['covered_skills_priority']}")
                    
                    if isinstance(project['highlights'], list):
                        print("   Highlights:")
                        for highlight in project['highlights']:
                            print(f"   - {highlight}")
                    else:
                        print(f"   Highlights: {project['highlights']}")
                
                # Allow user to select projects
                while True:
                    try:
                        response = await aioconsole.ainput("\nEnter project numbers to include (comma-separated) or 'all' for all projects: ")
                        if response.lower() == 'all':
                            return phase_result
                        
                        selected_indices = [int(idx.strip()) for idx in response.split(',')]
                        if all(1 <= idx <= len(project_recommendations) for idx in selected_indices):
                            # Update project recommendations with selected projects only
                            selected_projects = [project_recommendations[idx-1] for idx in selected_indices]
                            phase_result["project_recommendations"] = selected_projects
                            return phase_result
                        else:
                            print(f"Please enter numbers between 1 and {len(project_recommendations)}")
                    except ValueError:
                        print("Please enter valid numbers separated by commas")
                    
            return phase_result
            
        except Exception as e:
            self.logger.error(f"Error in user interaction for {phase_name} phase: {str(e)}")
            raise

async def main():
    """Main entry point for the CV customization system."""
    parser = argparse.ArgumentParser(description='Create a customized CV using AI agents')
    parser.add_argument(
        '--resume',
        type=str,
        required=True,
        help='Path to the input resume file'
    )
    parser.add_argument(
        '--job-description',
        type=str,
        required=True,
        help='Job description text or path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Path to save output'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='src/config',
        help='Directory containing YAML config files'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode'
    )

    args = parser.parse_args()

    try:
        # Initialize text processor
        text_processor = TextExtractionTool()
        
        # Process files asynchronously
        async def process_file(file_path: Path) -> str:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return await asyncio.to_thread(text_processor.run, file_path)

        # Process resume and job description concurrently
        resume_path = Path(args.resume)
        job_path = Path(args.job_description)
        
        resume_text, job_description = await asyncio.gather(
            process_file(resume_path),
            process_file(job_path) if job_path.exists() else asyncio.sleep(0, args.job_description)
        )

        # Initialize system
        system = CVCustomizationSystem(args.config_dir)
        await system.initialize()  # Complete async initialization
        
        output_path = Path(args.output)
        
        # Process resume
        result = await system.process_resume(
            resume_text=resume_text,
            job_description=job_description,
            output_path=output_path,
            interactive=not args.non_interactive
        )
        
        print("Processing completed successfully!")
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)