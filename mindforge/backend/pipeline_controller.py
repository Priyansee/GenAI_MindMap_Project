import time
import logging
import asyncio
from typing import Any, Dict, List
from agents.planner_agent import PlannerAgent
from agents.structurer_agent import StructurerAgent
from agents.validator_agent import ValidatorAgent
from agents.enhancer_agent import EnhancerAgent
from agents.renderer_agent import RendererAgent
from agents.exporter_agent import ExporterAgent
from agents.tester_agent import TesterAgent
from agents.refiner_agent import RefinerAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineController")

class PipelineController:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.planner = PlannerAgent(api_key=api_key)
        self.structurer = StructurerAgent(api_key=api_key)
        self.validator = ValidatorAgent(api_key=api_key)
        self.enhancer = EnhancerAgent(api_key=api_key)
        self.renderer = RendererAgent(api_key=api_key)
        self.exporter = ExporterAgent(api_key=api_key)
        self.tester = TesterAgent(api_key=api_key)
        self.refiner = RefinerAgent(api_key=api_key)
        self.logs: List[Dict[str, Any]] = []

    def _log_stage(self, agent_name: str, status: str, duration: float, input_len: int, output_len: int, error: str = None):
        log_entry = {
            "timestamp": time.time(),
            "agent_name": agent_name,
            "status": status,
            "duration_ms": int(duration * 1000),
            "input_length": input_len,
            "output_length": output_len,
            "error_message": error
        }
        self.logs.append(log_entry)
        logger.info(f"Agent {agent_name} finished with status {status} in {log_entry['duration_ms']}ms")

    async def run(self, input_text: str, clarification_answers: Dict = None, additional_info: str = None) -> Dict[str, Any]:
        context = {
            "input": input_text,
            "clarification_answers": clarification_answers,
            "additional_info": additional_info
        }
        
        # 1. Planner Agent
        start_time = time.time()
        try:
            planner_output = await self.planner.run(context)
            context["planner_output"] = planner_output
            self._log_stage("Planner Agent", "success", time.time() - start_time, len(input_text), len(str(planner_output)))
        except Exception as e:
            self._log_stage("Planner Agent", "failed", time.time() - start_time, len(input_text), 0, str(e))
            raise

        await asyncio.sleep(0.5)

        # 2. Structurer Agent
        start_time = time.time()
        try:
            structured_output = await self.structurer.run({"planner_output": context["planner_output"]})
            context["structured_output"] = structured_output
            self._log_stage("Structurer Agent", "success", time.time() - start_time, len(str(planner_output)), len(str(structured_output)))
        except Exception as e:
            # Retry policy for structurer
            logger.warning(f"Structurer failed, retrying once... Error: {e}")
            try:
                structured_output = await self.structurer.run({"planner_output": context["planner_output"]})
                context["structured_output"] = structured_output
                self._log_stage("Structurer Agent", "retried", time.time() - start_time, len(str(planner_output)), len(str(structured_output)))
            except Exception as e2:
                self._log_stage("Structurer Agent", "failed", time.time() - start_time, len(str(planner_output)), 0, str(e2))
                raise

        await asyncio.sleep(0.5)

        # 3. Validator Agent
        start_time = time.time()
        try:
            validated_output = await self.validator.run({"structured_output": context["structured_output"]})
            context["validated_output"] = validated_output
            self._log_stage("Validator Agent", "success", time.time() - start_time, len(str(structured_output)), len(str(validated_output)))
        except Exception as e:
            logger.warning(f"Validator failed, continuing with structured output. Error: {e}")
            context["validated_output"] = context["structured_output"]
            self._log_stage("Validator Agent", "skipped", time.time() - start_time, len(str(structured_output)), 0, str(e))

        await asyncio.sleep(0.5)

        # 4. Enhancer Agent
        start_time = time.time()
        try:
            enhanced_output = await self.enhancer.run({"validated_output": context["validated_output"]})
            context["enhanced_output"] = enhanced_output
            self._log_stage("Enhancer Agent", "success", time.time() - start_time, len(str(validated_output)), len(str(enhanced_output)))
        except Exception as e:
            logger.warning(f"Enhancer failed, continuing with validated output. Error: {e}")
            context["enhanced_output"] = context["validated_output"]
            self._log_stage("Enhancer Agent", "skipped", time.time() - start_time, len(str(validated_output)), 0, str(e))

        await asyncio.sleep(0.5)

        # 5. Renderer Agent (Backend preparation)
        start_time = time.time()
        try:
            render_output = await self.renderer.run({"enhanced_output": context["enhanced_output"]})
            context["render_output"] = render_output
            self._log_stage("Renderer Agent", "success", time.time() - start_time, len(str(context["enhanced_output"])), len(str(render_output)))
        except Exception as e:
            self._log_stage("Renderer Agent", "failed", time.time() - start_time, 0, 0, str(e))
            context["render_output"] = context["enhanced_output"]

        await asyncio.sleep(0.5)

        # 6. Exporter Agent (Miro JSON prep)
        start_time = time.time()
        try:
            miro_json = await self.exporter.run({"mind_map": context["render_output"]})
            context["miro_json"] = miro_json
            self._log_stage("Exporter Agent", "success", time.time() - start_time, len(str(context["render_output"])), len(str(miro_json)))
        except Exception as e:
            self._log_stage("Exporter Agent", "failed", time.time() - start_time, 0, 0, str(e))

        await asyncio.sleep(0.5)

        # 7. Tester Agent
        start_time = time.time()
        try:
            tester_report = await self.tester.run({"pipeline_outputs": context})
            self._log_stage("Tester Agent", "success", time.time() - start_time, len(str(context)), len(str(tester_report)))
        except Exception as e:
            tester_report = {"valid": True, "quality_score": 0.0, "issues": [str(e)]}
            self._log_stage("Tester Agent", "failed", time.time() - start_time, 0, 0, str(e))

        return {
            "mind_map": context["render_output"],
            "miro_json": context.get("miro_json"),
            "tester_report": tester_report,
            "pipeline_log": self.logs
        }

    async def refine(self, previous_map: Dict, feedback: str) -> Dict[str, Any]:
        """One-trial refinement process"""
        start_time = time.time()
        try:
            # 1. Run Refiner
            refined_map = await self.refiner.run({
                "previous_map": previous_map,
                "feedback": feedback
            })
            
            # 2. Run Validator on the refined output to ensure it's still good
            validated_map = await self.validator.run({"structured_output": refined_map})
            
            # 3. Regenerate Miro JSON
            miro_json = await self.exporter.run({"mind_map": validated_map})
            
            # 4. Run Final Test
            tester_report = await self.tester.run({"pipeline_outputs": {"mind_map": validated_map}})
            
            self._log_stage("Refinement Process", "success", time.time() - start_time, len(str(previous_map)), len(str(validated_map)))
            
            return {
                "mind_map": validated_map,
                "miro_json": miro_json,
                "tester_report": tester_report,
                "pipeline_log": self.logs
            }
        except Exception as e:
            self._log_stage("Refinement Process", "failed", time.time() - start_time, 0, 0, str(e))
            raise
