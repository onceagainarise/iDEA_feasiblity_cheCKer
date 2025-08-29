"""
AI Innovation Feasibility Agent using LangGraph and MCP
Complete implementation for evaluating technological feasibility of innovative ideas
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import json
from dataclasses import dataclass, asdict

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import Groq for LLM
from groq import AsyncGroq

# Import our custom modules
from config import settings
from mcp_integrationsp import RealMCPTools

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    """Structure for research findings"""
    papers: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    patents: List[Dict[str, Any]]
    github_projects: List[Dict[str, Any]]
    timestamp: str

@dataclass
class FeasibilityAnalysis:
    """Final feasibility analysis structure"""
    idea_summary: str
    current_status: str
    barriers: List[str]
    incremental_steps: List[str]
    future_outlook: str
    timeline_estimate: str
    conclusion: str  # "not feasible", "partially feasible", "feasible now", "feasible in future"
    confidence_score: float

class AgentState(TypedDict):
    """State maintained throughout the agent workflow"""
    messages: List[AnyMessage]
    user_idea: str
    research_query: str
    research_results: Optional[ResearchResult]
    analysis_draft: Optional[FeasibilityAnalysis]
    iteration_count: int
    needs_more_research: bool
    final_report: Optional[str]

class InnovationFeasibilityAgent:
    """Main agent class using LangGraph for orchestration"""
    
    def __init__(self):
        self.llm = AsyncGroq(api_key=settings.groq_api_key)
        self.mcp_tools = RealMCPTools(settings)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_idea", self.parse_idea)
        workflow.add_node("conduct_research", self.conduct_research)
        workflow.add_node("analyze_feasibility", self.analyze_feasibility)
        workflow.add_node("reflect_and_iterate", self.reflect_and_iterate)
        workflow.add_node("generate_final_report", self.generate_final_report)
        
        # Define the workflow flow
        workflow.add_edge(START, "parse_idea")
        workflow.add_edge("parse_idea", "conduct_research")
        workflow.add_edge("conduct_research", "analyze_feasibility")
        workflow.add_edge("analyze_feasibility", "reflect_and_iterate")
        
        # Conditional edges based on iteration needs
        workflow.add_conditional_edges(
            "reflect_and_iterate",
            self._should_continue_research,
            {
                "continue": "conduct_research",
                "finish": "generate_final_report"
            }
        )
        
        workflow.add_edge("generate_final_report", END)
        
        # Set up memory for conversation history
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _should_continue_research(self, state: AgentState) -> str:
        """Decide whether to continue research or finish"""
        if (state["needs_more_research"] and 
            state["iteration_count"] < settings.max_research_iterations):
            return "continue"
        return "finish"
    
    async def parse_idea(self, state: AgentState) -> Dict[str, Any]:
        """Parse and understand the user's idea"""
        logger.info("Parsing user idea...")
        
        user_idea = state["user_idea"]
        
        prompt = f"""
        Analyze this innovative idea and extract key research topics:
        
        Idea: "{user_idea}"
        
        Tasks:
        1. Identify the core technological components
        2. Determine what research areas are most relevant
        3. Generate 3-5 focused search queries for research
        
        Return your analysis as JSON with these fields:
        - core_components: List of main technologies involved
        - research_areas: List of relevant scientific/technical fields
        - search_queries: List of specific queries for research
        """
        
        try:
            response = await self.llm.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content
            # Simple JSON extraction (in production, use more robust parsing)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                research_query = " ".join(analysis.get("search_queries", [user_idea]))
            else:
                research_query = user_idea
            
            return {
                **state,
                "research_query": research_query,
                "iteration_count": 0
            }
            
        except Exception as e:
            logger.error(f"Idea parsing failed: {e}")
            return {
                **state,
                "research_query": user_idea,
                "iteration_count": 0
            }
    
    async def conduct_research(self, state: AgentState) -> Dict[str, Any]:
        """Conduct parallel research across multiple sources"""
        logger.info(f"Conducting research for: {state['research_query']}")
        
        query = state["research_query"]
        
        # Parallel research across multiple sources
        research_tasks = [
            self.mcp_tools.search_arxiv(query, max_results=15),
            self.mcp_tools.search_news(query, days=90),
            self.mcp_tools.search_patents(query),
            self.mcp_tools.get_github_projects(query, limit=10)
        ]
        
        try:
            results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            research_result = ResearchResult(
                papers=results[0] if not isinstance(results[0], Exception) else [],
                news=results[1] if not isinstance(results[1], Exception) else [],
                patents=results[2] if not isinstance(results[2], Exception) else [],
                github_projects=results[3] if not isinstance(results[3], Exception) else [],
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Research completed: {len(research_result.papers)} papers, "
                       f"{len(research_result.news)} news articles, "
                       f"{len(research_result.patents)} patents, "
                       f"{len(research_result.github_projects)} GitHub projects")
            
            return {
                **state,
                "research_results": research_result
            }
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return state
    
    async def analyze_feasibility(self, state: AgentState) -> Dict[str, Any]:
        """Analyze feasibility based on research findings"""
        logger.info("Analyzing feasibility...")
        
        user_idea = state["user_idea"]
        research = state.get("research_results")
        
        if not research:
            logger.warning("No research results available for analysis")
            return state
        
        # Prepare research context
        research_context = self._format_research_context(research)
        
        analysis_prompt = f"""
        You are an expert technology feasibility analyst. Based on the research findings, analyze this innovative idea:
        
        IDEA: "{user_idea}"
        
        RESEARCH FINDINGS:
        {research_context}
        
        Provide a comprehensive feasibility analysis with these sections:
        
        1. IDEA SUMMARY (2-3 sentences explaining the concept simply)
        
        2. CURRENT TECHNOLOGICAL STATUS (what exists today that's related)
        
        3. BARRIERS AND CHALLENGES (technical, economic, regulatory, physical limits)
        
        4. INCREMENTAL STEPS (specific steps that could make this more feasible)
        
        5. FUTURE OUTLOOK (what breakthroughs would be needed, realistic timeline)
        
        6. CONCLUSION (choose one):
           - "not feasible" - physically impossible or extremely impractical
           - "partially feasible" - some aspects possible, others not
           - "feasible now" - technically possible with current technology
           - "feasible in future" - not now, but likely possible with future advances
        
        7. CONFIDENCE SCORE (0.0 to 1.0 based on research quality and clarity)
        
        Be honest, evidence-based, and cite specific research findings where relevant.
        """
        
        try:
            response = await self.llm.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract structured analysis (simplified parsing)
            analysis = self._parse_analysis_response(analysis_text)
            
            return {
                **state,
                "analysis_draft": analysis
            }
            
        except Exception as e:
            logger.error(f"Feasibility analysis failed: {e}")
            return state
    
    async def reflect_and_iterate(self, state: AgentState) -> Dict[str, Any]:
        """Self-reflection to determine if more research is needed"""
        logger.info("Reflecting on analysis quality...")
        
        analysis = state.get("analysis_draft")
        research = state.get("research_results")
        iteration = state["iteration_count"]
        
        if not analysis or not research:
            return {
                **state,
                "needs_more_research": False,
                "iteration_count": iteration + 1
            }
        
        reflection_prompt = f"""
        Review this feasibility analysis and determine if more research is needed:
        
        CURRENT ANALYSIS CONFIDENCE: {analysis.confidence_score}
        RESEARCH SOURCES: {len(research.papers)} papers, {len(research.news)} news, {len(research.patents)} patents
        ITERATION: {iteration + 1} of {settings.max_research_iterations}
        
        Key gaps to check:
        - Are there recent breakthrough papers missing?
        - Is the analysis lacking in specific technical details?
        - Are there contradictory findings that need resolution?
        - Is more domain-specific research needed?
        
        Respond with:
        - "SUFFICIENT" if analysis is comprehensive enough
        - "NEED_MORE: [specific research focus]" if more research needed
        """
        
        try:
            response = await self.llm.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.1
            )
            
            reflection = response.choices[0].message.content.strip()
            needs_more = reflection.startswith("NEED_MORE")
            
            # Update research query if more research needed
            new_query = state["research_query"]
            if needs_more and ":" in reflection:
                new_query = reflection.split(":")[1].strip()
            
            return {
                **state,
                "needs_more_research": needs_more,
                "iteration_count": iteration + 1,
                "research_query": new_query
            }
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {
                **state,
                "needs_more_research": False,
                "iteration_count": iteration + 1
            }
    
    async def generate_final_report(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final structured feasibility report"""
        logger.info("Generating final feasibility report...")
        
        analysis = state.get("analysis_draft")
        if not analysis:
            return {
                **state,
                "final_report": "Analysis could not be completed due to insufficient data."
            }
        
        # Generate structured final report
        report_template = f"""
# Innovation Feasibility Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 💡 Idea Summary
{analysis.idea_summary}

## 🔬 Current Technological Status
{analysis.current_status}

## 🚧 Barriers and Challenges
{self._format_list(analysis.barriers)}

## 📈 Incremental Development Steps
{self._format_list(analysis.incremental_steps)}

## 🔮 Future Outlook and Timeline
{analysis.future_outlook}

**Estimated Timeline:** {analysis.timeline_estimate}

## ⚖️ Final Assessment
**Conclusion:** {analysis.conclusion.upper()}

**Confidence Level:** {analysis.confidence_score:.1%}

---
*This analysis is based on current research and technological understanding as of {datetime.now().strftime('%B %Y')}*
        """
        
        return {
            **state,
            "final_report": report_template.strip()
        }
    
    def _format_research_context(self, research: ResearchResult) -> str:
        """Format research results for LLM context"""
        context_parts = []
        
        # Recent papers
        if research.papers:
            papers_text = "\n".join([
                f"- {p['title']} ({p['year']}) - {p['abstract'][:200]}..."
                for p in research.papers[:5]
            ])
            context_parts.append(f"RECENT PAPERS:\n{papers_text}")
        
        # News developments
        if research.news:
            news_text = "\n".join([
                f"- {n['title']} ({n['date']}) - {n['description'][:150]}..."
                for n in research.news[:3]
            ])
            context_parts.append(f"RECENT NEWS:\n{news_text}")
        
        # Patents
        if research.patents:
            patent_text = "\n".join([
                f"- {p['title']} by {p['inventor']} ({p['publication_date']})"
                for p in research.patents[:3]
            ])
            context_parts.append(f"RELEVANT PATENTS:\n{patent_text}")
        
        # GitHub projects
        if research.github_projects:
            github_text = "\n".join([
                f"- {g['name']} ({g['stars']} stars) - {g['description'][:100]}..."
                for g in research.github_projects[:3]
            ])
            context_parts.append(f"OPEN SOURCE PROJECTS:\n{github_text}")
        
        return "\n\n".join(context_parts)
    
    def _parse_analysis_response(self, text: str) -> FeasibilityAnalysis:
        """Parse LLM response into structured analysis"""
        # Simple parsing logic - in production, use more robust extraction
        sections = {
            'idea_summary': '',
            'current_status': '',
            'barriers': [],
            'incremental_steps': [],
            'future_outlook': '',
            'timeline_estimate': 'Unknown',
            'conclusion': 'partially feasible',
            'confidence_score': 0.5
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'IDEA SUMMARY' in line.upper():
                current_section = 'idea_summary'
            elif 'CURRENT TECHNOLOGICAL STATUS' in line.upper():
                current_section = 'current_status'
            elif 'BARRIERS' in line.upper():
                current_section = 'barriers'
            elif 'INCREMENTAL STEPS' in line.upper():
                current_section = 'incremental_steps'
            elif 'FUTURE OUTLOOK' in line.upper():
                current_section = 'future_outlook'
            elif 'CONCLUSION' in line.upper():
                current_section = 'conclusion'
            elif 'CONFIDENCE' in line.upper():
                current_section = 'confidence'
            elif line and current_section:
                if current_section in ['barriers', 'incremental_steps']:
                    if line.startswith('-') or line.startswith('•'):
                        sections[current_section].append(line[1:].strip())
                    elif line:
                        sections[current_section].append(line)
                elif current_section == 'conclusion':
                    if any(conclusion in line.lower() for conclusion in ['not feasible', 'partially feasible', 'feasible now', 'feasible in future']):
                        for conclusion in ['not feasible', 'partially feasible', 'feasible now', 'feasible in future']:
                            if conclusion in line.lower():
                                sections['conclusion'] = conclusion
                                break
                elif current_section == 'confidence':
                    import re
                    numbers = re.findall(r'0\.\d+|\d+\.\d+%?', line)
                    if numbers:
                        score = float(numbers[0].replace('%', '')) 
                        sections['confidence_score'] = score if score <= 1.0 else score / 100
                else:
                    sections[current_section] = sections[current_section] + ' ' + line if sections[current_section] else line
        
        return FeasibilityAnalysis(**sections)
    
    def _format_list(self, items: List[str]) -> str:
        """Format list items for report"""
        if not items:
            return "None identified"
        return "\n".join([f"• {item}" for item in items])
    
    async def process_idea(self, user_idea: str) -> str:
        """Main entry point to process an innovative idea"""
        logger.info(f"Starting feasibility analysis for: {user_idea[:100]}...")
        
        # Initial state
        initial_state = {
            "messages": [],
            "user_idea": user_idea,
            "research_query": "",
            "research_results": None,
            "analysis_draft": None,
            "iteration_count": 0,
            "needs_more_research": True,
            "final_report": None
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"analysis_{datetime.now().timestamp()}"}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)
            return final_state.get("final_report", "Analysis could not be completed.")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return f"Analysis failed due to technical error: {str(e)}"
        finally:
            await self.mcp_tools.close()

# CLI Interface
class FeasibilityAgentCLI:
    """Command-line interface for the agent"""
    
    def __init__(self):
        self.agent = InnovationFeasibilityAgent()
    
    async def run_interactive(self):
        """Run interactive mode"""
        print("🤖 AI Innovation Feasibility Agent")
        print("=" * 50)
        print("Enter innovative ideas for feasibility analysis.")
        print("Type 'quit' to exit, 'help' for examples.\n")
        
        while True:
            try:
                user_input = input("\n💡 Enter your innovative idea: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    self._show_examples()
                    continue
                
                if not user_input:
                    continue
                
                print("\n🔍 Analyzing feasibility... (this may take a minute)")
                
                report = await self.agent.process_idea(user_input)
                print("\n" + "="*60)
                print(report)
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_examples(self):
        """Show example ideas"""
        examples = [
            "Replace VLBI with satellite constellation for space interferometry",
            "Use quantum entanglement for instantaneous global communication",
            "Build space elevator using carbon nanotube cables",
            "Create fusion-powered atmospheric processors for terraforming Mars",
            "Develop brain-computer interface for direct knowledge transfer",
            "Use AI-controlled swarm robotics for asteroid mining",
        ]
        
        print("\n📝 Example Ideas:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print()

# Batch processing mode
async def process_ideas_batch(ideas_file: str, output_file: str):
    """Process multiple ideas from a file"""
    agent = InnovationFeasibilityAgent()
    
    try:
        with open(ideas_file, 'r') as f:
            ideas = [line.strip() for line in f if line.strip()]
        
        results = []
        for i, idea in enumerate(ideas, 1):
            print(f"Processing idea {i}/{len(ideas)}: {idea[:50]}...")
            
            report = await agent.process_idea(idea)
            results.append({
                'idea': idea,
                'report': report,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Batch processing complete! Results saved to {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{ideas_file}' not found")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

# Main execution
async def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 2:
        # Batch mode
        ideas_file = sys.argv[1]
        output_file = sys.argv[2]
        await process_ideas_batch(ideas_file, output_file)
    
    elif len(sys.argv) == 2:
        # Single idea mode
        idea = sys.argv[1]
        agent = InnovationFeasibilityAgent()
        report = await agent.process_idea(idea)
        print(report)
    
    else:
        # Interactive mode
        cli = FeasibilityAgentCLI()
        await cli.run_interactive()

if __name__ == "__main__":
    # Ensure we have required environment variables
    try:
        settings.groq_api_key  # This will raise an error if not set
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Setup Error: {e}")
        print("\nMake sure you have:")
        print("1. GROQ_API_KEY set in your .env file")
        print("2. GROQ_MODEL set in your .env file (e.g., 'llama-3.1-70b-versatile')")
        print("3. Installed dependencies: pip install langgraph groq httpx pydantic-settings")