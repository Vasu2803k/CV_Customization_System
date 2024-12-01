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

class CVCustomizationSystem:
    """A system for customizing CVs using multiple specialized agents with fallback mechanisms."""
    
    async def __init__(self, config_path: str):
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

            # Load configuration
            self.config = await self._load_config(config_path)
            
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
            
            # Initialize LLM configurations
            self.llm_config = self.config["llm_config"]["openai_config"]
            self.fallback_config = self.config["llm_config"]["fallback_config"]
            
            # Initialize agents and group chats
            self.agents = await self._initialize_agents()
            self.group_chats = await self._initialize_group_chats()
            
            self.logger.info("CV Customization System initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CV Customization System: {str(e)}")

    async def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from YAML files."""
        try:
            config_dir = Path(config_path).parent
            
            # Load config files synchronously
            agents_config = self._load_yaml(config_dir / 'agents.yaml')
            templates_config = self._load_yaml(config_dir / 'templates.yaml')
            projects_config = self._load_yaml(config_dir / 'projects.yaml')
            
            # Validate LLM configurations
            required_llm_configs = ['openai_config', 'claude_config', 'fallback_config']
            for config in required_llm_configs:
                if config not in agents_config.get('llm_config', {}):
                    raise ValueError(f"Missing required LLM configuration: {config}")
            
            # Validate template configuration
            if not templates_config.get('templates'):
                raise ValueError("No LaTeX templates found in configuration")
            
            return {
                "agents": agents_config["agents"],
                "llm_config": agents_config["llm_config"],
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
        
        # Base LLM configurations with fallback chain
        llm_configs = {
            "primary": self.config["llm_config"]["openai_config"],
            "secondary": self.config["llm_config"]["claude_config"],
            "fallback": self.config["llm_config"]["fallback_config"]
        }
        
        # Core agents with prompts and schemas from config
        core_agents = {
            "analyzer": {
                "role": self.config["agents"]["resume_analyzer"]["role"],
                "system_prompt": self.config["agents"]["resume_analyzer"]["system_prompt"],
                "output_format": self.config["agents"]["resume_analyzer"]["output_format"],
                "output_schema": self.config["agents"]["resume_analyzer"]["output_schema"]
            },
            "optimizer": {
                "role": self.config["agents"]["resume_optimizer"]["role"],
                "system_prompt": self.config["agents"]["resume_optimizer"]["system_prompt"],
                "output_format": self.config["agents"]["resume_optimizer"]["output_format"],
                "output_schema": self.config["agents"]["resume_optimizer"]["output_schema"]
            },
            "formatter": {
                "role": self.config["agents"]["resume_formatter"]["role"],
                "system_prompt": self.config["agents"]["resume_formatter"]["system_prompt"],
                "output_format": self.config["agents"]["resume_formatter"]["output_format"]
            }
        }

        # Initialize agents with fallback chain
        for role, config in core_agents.items():
            try:
                # Construct system message with output requirements
                system_message = (
                    f"{config['system_prompt']}\n\n"
                    f"Output Format: {config['output_format']}\n"
                )
                
                # Add schema if present
                if 'output_schema' in config:
                    system_message += f"Output Schema: {config['output_schema']}\n"

                # Primary agent (GPT-4)
                primary_agent = AssistantAgent(
                    name=f"{role}_primary",
                    system_message=system_message,
                    llm_config=llm_configs["primary"]
                )
                
                # Secondary agent (Claude)
                secondary_agent = AssistantAgent(
                    name=f"{role}_secondary",
                    system_message=system_message,
                    llm_config=llm_configs["secondary"]
                )
                
                # Fallback agent
                fallback_agent = AssistantAgent(
                    name=f"{role}_fallback",
                    system_message=system_message,
                    llm_config=llm_configs["fallback"]
                )
                
                agents[role] = {
                    "primary": primary_agent,
                    "secondary": secondary_agent,
                    "fallback": fallback_agent
                }
                
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {role}: {str(e)}")
                raise

        # Initialize user proxy agent with config
        try:
            user_proxy_config = self.config["agents"]["user_proxy"]
            agents["user_proxy"] = UserProxyAgent(
                name="user_proxy",
                system_message=user_proxy_config["system_prompt"],
                human_input_mode="ALWAYS",
                code_execution_config=False
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize user proxy agent: {str(e)}")
            raise

        return agents

    async def _initialize_group_chats(self) -> Dict[str, GroupChatManager]:
        """Initialize streamlined group chats."""
        chat_configs = {
            "analysis": {
                "agents": ["user_proxy", "analyzer", "evaluator"],
                "max_round": 1,
                "description": "Resume analysis and scoring",
                "manager_system_message": "Guide the analysis process to generate clear scoring and recommendations."
            },
            "optimization": {
                "agents": ["user_proxy", "optimizer", "evaluator"],
                "max_round": 3,
                "description": "Resume optimization",
                "manager_system_message": "Guide the optimization process to enhance content and suggest improvements."
            },
            "formatting": {
                "agents": ["user_proxy", "formatter", "evaluator"],
                "max_round": 3,
                "description": "Resume formatting",
                "manager_system_message": "Guide the formatting process to ensure professional presentation."
            }
        }

        group_chats = {}
        
        for chat_name, config in chat_configs.items():
            try:
                # Create group chat
                chat_agents = []
                for agent_name in config["agents"]:
                    if agent_name == "user_proxy":
                        chat_agents.append(self.agents[agent_name])
                    else:
                        chat_agents.append(self.agents[agent_name]["primary"])
                
                chat = GroupChat(
                    agents=chat_agents,
                    messages=[],
                    max_round=config["max_round"],
                    description=config["description"],
                    speaker_selection_method="auto"
                )
                
                # Create manager
                manager = GroupChatManager(
                    groupchat=chat,
                    llm_config=self.llm_config,
                    system_message=config["manager_system_message"]
                )
                
                group_chats[chat_name] = manager
                
            except Exception as e:
                self.logger.error(f"Failed to initialize group chat {chat_name}: {str(e)}")
                raise

        return group_chats

    async def _run_group_chat_with_fallback(
        self,
        chat_name: str,
        prompt: str,
        message_id: str,
        max_retries: int = 3
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
                    
                # Try secondary LLM (Claude)
                elif retry_count == 1:
                    self.logger.info(f"Attempting secondary LLM (Claude) for {chat_name}")
                    result = await self._run_group_chat(
                        chat_name,
                        prompt,
                        message_id,
                        "secondary"
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

    async def _run_group_chat(
        self,
        chat_name: str,
        prompt: str,
        message_id: str,
        llm_type: str = "primary"
    ) -> Dict:
        """Run a specific group chat with the given LLM type."""
        try:
            # Add template and project information to formatting phase
            if chat_name == "formatting":
                # Load template information
                template_info = self.config.get("templates", {})
                prompt = f"""
                Format optimized resume using available LaTeX templates:

                Available Templates:
                {json.dumps(template_info, indent=2)}

                Optimized Content:
                {prompt}
                """

            # Add project suggestions to analysis phase
            elif chat_name == "analysis":
                # Load project information
                project_info = self.config.get("projects", {})
                prompt = f"""
                Analyze resume and job description, including relevant project suggestions:

                Available Project Templates:
                {json.dumps(project_info, indent=2)}

                Analysis Request:
                {prompt}
                """

            # Get the appropriate agents based on LLM type
            chat_agents = []
            for agent_name in self.group_chats[chat_name].groupchat.agents:
                if agent_name == "user_proxy":
                    chat_agents.append(self.agents["user_proxy"])
                else:
                    # Get the appropriate agent based on the phase
                    if chat_name == "analysis":
                        agent = self.agents["analyzer"][llm_type]
                    elif chat_name == "optimization":
                        agent = self.agents["optimizer"][llm_type]
                    elif chat_name == "formatting":
                        agent = self.agents["formatter"][llm_type]
                    else:
                        raise ValueError(f"Invalid chat name: {chat_name}")
                    chat_agents.append(agent)
            
            # Create group chat with appropriate configuration
            chat = GroupChat(
                agents=chat_agents,
                messages=[],
                max_round=self.group_chats[chat_name].groupchat.max_round,
                description=self.group_chats[chat_name].groupchat.description
            )
            
            # Create manager with appropriate LLM config
            llm_config = (
                self.llm_config if llm_type == "primary"
                else self.fallback_config if llm_type == "fallback"
                else self.config["llm_config"]["claude_config"]
            )
            
            manager = GroupChatManager(
                groupchat=chat,
                llm_config=llm_config,
                system_message=self.group_chats[chat_name].system_message
            )
            
            # Run the chat and return results
            start_time = time.time()
            result = await manager.run(prompt, message_id=message_id)
            
            # Track metrics
            await self._track_step_metrics(
                step=f"{chat_name}_{llm_type}",
                start_time=start_time,
                result=result
            )
            
            return result
            
        except Exception as e:
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
                {json.dumps(analysis_result, indent=2)}""",
                "optimization_phase"
            )

            if interactive:
                optimization_result = await self._handle_user_interaction(
                    optimization_result,
                    "Optimization"
                )

            # Phase 3: Formatting with fallback and template selection
            formatting_prompt = f"""Format optimized resume:

            Available Templates:
            {json.dumps(self.config['templates'], indent=2)}

            Optimized Content:
            {json.dumps(optimization_result, indent=2)}

            Selected Projects:
            {json.dumps(optimization_result.get('selected_projects', []), indent=2)}
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


    async def _save_output(self, output: Dict, output_path: Path):
        """Save output with metrics."""
        try:
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metrics to output
            output["metadata"]["metrics"] = {
                "step_timings": self.metrics["steps"],
                "total_duration": self.metrics["cumulative"]["total_duration"],
                "total_tokens": self.metrics["cumulative"]["total_tokens"],
                "total_cost": self.metrics["cumulative"]["total_cost"],
                "total_retries": self.metrics["cumulative"]["total_retries"]
            }
            
            # Save output synchronously
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            self.logger.info(f"Output saved to {output_path}")
            self.logger.info("\nFinal Metrics:")
            self.logger.info(f"Total Duration: {output['metadata']['metrics']['total_duration']:.2f}s")
            self.logger.info(f"Total Tokens: {output['metadata']['metrics']['total_tokens']}")
            self.logger.info(f"Total Cost: ${output['metadata']['metrics']['total_cost']:.4f}")
            self.logger.info(f"Total Retries: {output['metadata']['metrics']['total_retries']}")
            
        except Exception as e:
            self.logger.error(f"Error saving output: {str(e)}")
            raise

    async def _handle_user_interaction(self, result: Dict, phase: str) -> Dict:
        """Enhanced user interaction with detailed feedback using logging."""
        try:
            self.logger.info(f"=== {phase} Results and Suggestions ===")
            
            # Log key results based on phase
            if phase == "Analysis":
                self.logger.info("\nSkill Match Analysis:")
                for skill, score in result.get("skill_matches", {}).items():
                    self.logger.info(f"- {skill}: {score}%")
                
                self.logger.info("\nExperience Alignment:")
                for exp in result.get("experience_alignment", []):
                    self.logger.info(f"- {exp['role']}: {exp['match_score']}% match")
                
                # Add project suggestions
                if "project_suggestions" in result:
                    self.logger.info("\nRecommended Projects:")
                    for project in result["project_suggestions"]:
                        self.logger.info(f"\n- {project['name']}")
                        self.logger.info(f"  Relevance: {project['relevance_score']}%")
                        self.logger.info(f"  Description: {project['description']}")
            
            # Log options with additional project selection option for Analysis phase
            self.logger.info("\nOptions:")
            self.logger.info("1. Accept and proceed")
            self.logger.info("2. Request modifications")
            self.logger.info("3. Show detailed suggestions")
            self.logger.info("4. Show performance metrics")
            if phase == "Analysis":
                self.logger.info("5. Select recommended projects")
            
            choice = await self._get_user_input("\nEnter your choice (1-5): ")
            
            if choice == "5" and phase == "Analysis":
                return await self._handle_project_selection(result)
            elif choice == "1":
                self.logger.info("Proceeding with current results")
                return result
            elif choice == "2":
                self.logger.info("Requesting modifications")
                return await self._handle_modifications(result, phase)
            elif choice == "3":
                self.logger.info("Showing detailed suggestions")
                await self._show_detailed_suggestions(result, phase)
                return await self._handle_user_interaction(result, phase)
            elif choice == "4":
                self.logger.info("Showing performance metrics")
                await self._show_performance_metrics(phase)
                return await self._handle_user_interaction(result, phase)
            else:
                self.logger.warning("Invalid choice received")
                return await self._handle_user_interaction(result, phase)
            
        except Exception as e:
            self.logger.error(f"Error in user interaction: {str(e)}")
            raise

    async def _handle_modifications(self, result: Dict, phase: str) -> Dict:
        """Handle modifications to specific resume sections."""
        modifications = {}
        
        for section in result:
            if section not in ["analysis", "optimization", "formatting"]:
                print(f"Warning: Section '{section}' not found in results")
                continue
            
            print(f"\nCurrent {section}:")
            print(json.dumps(result[section], indent=2))
            
            try:
                modification = await self._get_user_input(
                    f"\nEnter modifications for {section} (JSON format):\n"
                )
                modifications[section] = json.loads(modification)
            except json.JSONDecodeError:
                print("Invalid JSON format. Skipping this section.")
                continue
            
        return modifications

    async def _show_detailed_suggestions(self, result: Dict, phase: str):
        """Show detailed suggestions based on phase."""
        print(f"\n=== Detailed Suggestions for {phase} ===")
        
        if phase == "Analysis":
            print("\nGap Analysis:")
            for gap in result.get("gaps", []):
                print(f"- {gap['description']}")
                print(f"  Suggestion: {gap['suggestion']}")
        
        elif phase == "Optimization":
            print("\nSection-wise Suggestions:")
            for section, suggestions in result.get("section_suggestions", {}).items():
                print(f"\n{section}:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")

    async def _show_performance_metrics(self, phase: str):
        """Show performance metrics for the current phase."""
        try:
            metrics = self.metrics["agent_performance"]
            phase_metrics = {
                agent: data["phases"].get(phase, {})
                for agent, data in metrics.items()
                if phase in data.get("phases", {})
            }
            
            print(f"\n=== Performance Metrics for {phase} ===")
            for agent, data in phase_metrics.items():
                print(f"\nAgent: {agent}")
                print(f"Calls: {data['calls']}")
                print(f"Average Duration: {data['average_duration']:.2f}s")
                print(f"Last Execution: {data['last_execution']}")
                
        except Exception as e:
            self.logger.error(f"Error showing performance metrics: {str(e)}")

    async def _get_user_input(self, prompt: str) -> str:
        """Get user input synchronously with timeout."""
        try:
            # Print prompt and get input synchronously
            print(prompt, end='', flush=True)
            
            # Use regular input with timeout using asyncio
            return await asyncio.wait_for(
                # asyncio.get_event_loop().run_in_executor(None, builtins.input),
                timeout=300  # 5 minutes
            )
        except asyncio.TimeoutError:
            raise TimeoutError("User input timed out after 5 minutes")
        except Exception as e:
            self.logger.error(f"Error getting user input: {str(e)}")
            raise

    async def _run_additional_analysis(
        self,
        phase: str,
        aspect: str,
        current_result: Dict
    ) -> Dict:
        """Run additional analysis on specific aspects."""
        try:
            analysis_prompts = {
                "Analysis": {
                    "skills": "Perform deeper analysis of skills alignment",
                    "experience": "Analyze experience relevance in detail",
                    "achievements": "Evaluate achievement impact and metrics",
                    "gaps": "Identify critical skill and experience gaps"
                },
                "Optimization": {
                    "bullets": "Analyze bullet point effectiveness",
                    "metrics": "Evaluate quantifiable metrics",
                    "formatting": "Check section formatting",
                    "balance": "Analyze content balance"
                },
                "Formatting": {
                    "layout": "Analyze layout effectiveness",
                    "spacing": "Check spacing and margins",
                    "consistency": "Verify formatting consistency",
                    "compatibility": "Test ATS compatibility"
                }
            }

            if phase not in analysis_prompts or aspect not in analysis_prompts[phase]:
                raise ValueError(f"Invalid analysis request: {phase} - {aspect}")

            prompt = f"""
            Perform additional analysis on the following aspect:
            Phase: {phase}
            Aspect: {aspect}
            
            Current Results:
            {json.dumps(current_result, indent=2)}
            
            Analysis Focus:
            {analysis_prompts[phase][aspect]}
            """

            result = await self._run_group_chat_with_fallback(
                phase.lower(),
                prompt,
                f"additional_analysis_{phase}_{aspect}"
            )

            return {
                "aspect": aspect,
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in additional analysis: {str(e)}")
            raise

    async def _track_step_metrics(self, step: str, start_time: float, result: Dict = None) -> None:
        """Track metrics for each processing step."""
        try:
            duration = time.time() - start_time
            timestamp = datetime.now().isoformat()

            if "steps" not in self.metrics:
                self.metrics["steps"] = {}

            # Track step-specific metrics
            step_metrics = {
                "duration": duration,
                "timestamp": timestamp,
                "token_usage": {
                    "prompt_tokens": result.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": result.get("token_usage", {}).get("completion_tokens", 0),
                    "total_tokens": result.get("token_usage", {}).get("total_tokens", 0),
                    "estimated_cost": self._calculate_token_cost(result.get("token_usage", {}))
                },
                "retries": self.metrics["retries"].get(step, 0)
            }

            self.metrics["steps"][step] = step_metrics
            self.metrics["phase_timings"][step] = duration

            # Log step completion
            self.logger.info(f"\nStep '{step}' completed:")
            self.logger.info(f"Duration: {duration:.2f}s")
            self.logger.info(f"Tokens used: {step_metrics['token_usage']['total_tokens']}")
            self.logger.info(f"Estimated cost: ${step_metrics['token_usage']['estimated_cost']:.4f}")
            
            # Update cumulative metrics
            if "cumulative" not in self.metrics:
                self.metrics["cumulative"] = {
                    "total_duration": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "total_retries": 0
                }

            cumulative = self.metrics["cumulative"]
            cumulative["total_duration"] += duration
            cumulative["total_tokens"] += step_metrics["token_usage"]["total_tokens"]
            cumulative["total_cost"] += step_metrics["token_usage"]["estimated_cost"]
            cumulative["total_retries"] += step_metrics["retries"]

            # Log cumulative metrics
            self.logger.info("\nCumulative Progress:")
            self.logger.info(f"Total time: {cumulative['total_duration']:.2f}s")
            self.logger.info(f"Total tokens: {cumulative['total_tokens']}")
            self.logger.info(f"Total cost: ${cumulative['total_cost']:.4f}")
            self.logger.info(f"Total retries: {cumulative['total_retries']}")

        except Exception as e:
            self.logger.error(f"Error tracking metrics: {str(e)}")

    def _calculate_token_cost(self, token_usage: Dict) -> float:
        """Calculate estimated cost based on token usage."""
        try:
            # Token cost rates (adjust as needed)
            rates = {
                "gpt-4": {
                    "prompt": 0.03,  # per 1K tokens
                    "completion": 0.06  # per 1K tokens
                },
                "gpt-3.5-turbo": {
                    "prompt": 0.0015,
                    "completion": 0.002
                }
            }
            
            model = token_usage.get("model", "gpt-3.5-turbo")
            rate = rates.get(model, rates["gpt-3.5-turbo"])
            
            prompt_cost = (token_usage.get("prompt_tokens", 0) / 1000) * rate["prompt"]
            completion_cost = (token_usage.get("completion_tokens", 0) / 1000) * rate["completion"]
            
            return prompt_cost + completion_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating token cost: {str(e)}")
            return 0.0

    async def _handle_project_selection(self, result: Dict) -> Dict:
        """Handle project selection from recommendations."""
        try:
            if "project_suggestions" not in result:
                self.logger.warning("No project suggestions available")
                return result
            
            self.logger.info("\nAvailable Projects:")
            for idx, project in enumerate(result["project_suggestions"], 1):
                self.logger.info(f"\n{idx}. {project['name']}")
                self.logger.info(f"   Relevance: {project['relevance_score']}%")
                self.logger.info(f"   Description: {project['description']}")
            
            selection = await self._get_user_input(
                "\nEnter project numbers to include (comma-separated): "
            )
            
            try:
                selected_indices = [int(i.strip()) - 1 for i in selection.split(",")]
                selected_projects = [
                    result["project_suggestions"][i] 
                    for i in selected_indices 
                    if 0 <= i < len(result["project_suggestions"])
                ]
                
                # Update result with selected projects
                result["selected_projects"] = selected_projects
                self.logger.info(f"\nSelected {len(selected_projects)} projects")
                
                return result
                
            except (ValueError, IndexError) as e:
                self.logger.error(f"Invalid project selection: {str(e)}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in project selection: {str(e)}")
            return result

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
            return asyncio.to_thread(text_processor.run, file_path)

        # Process resume and job description concurrently
        resume_path = Path(args.resume)
        job_path = Path(args.job_description)
        
        resume_text, job_description = await asyncio.gather(
            process_file(resume_path),
            process_file(job_path) if job_path.exists() else asyncio.sleep(0, args.job_description)
        )

        # Initialize system
        system = await CVCustomizationSystem(args.config_dir)
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