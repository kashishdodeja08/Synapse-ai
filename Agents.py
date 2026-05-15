# Agents.py
# Uses Groq (FREE) - Updated with working model

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ============================================
# STEP 1: Load API key
# ============================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    print("1. Create file .env in Synapse folder")
    print("2. Add: GROQ_API_KEY=gsk_your_key")
    exit(1)

print(f"Key loaded: {api_key[:15]}...")

# ============================================
# STEP 2: Create AI brain
# ============================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)

# ============================================
# AGENT 1: ARCHITECT
# ============================================
class ArchitectPlan(BaseModel):
    next_topic: str = Field(description="Exact topic name from syllabus")
    hours: int = Field(description="Study hours: 4 for remedial, 6 for new topic")
    remedial_needed: bool = Field(description="True if student failed previous quiz")
    reason: str = Field(description="Why this decision was made")

architect_parser = PydanticOutputParser(pydantic_object=ArchitectPlan)

architect_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an Academic Architect for BCA students at MLSU.
    You create study plans based on student progress.
    
    {format_instructions}
    
    Rules:
    - If mastery_level < 0.6 (60%), plan REMEDIAL for SAME topic (4 hours)
    - If mastery_level >= 0.6, plan NEXT topic in syllabus (6 hours)
    - Use exact topic names from syllabus
    - Always explain your reasoning"""),
    
    ("human", """Student progress:
    - Current Unit: {current_unit}
    - Current Topic Index: {current_topic}
    - Mastery Level: {mastery_level} (0.0 to 1.0)
    - Syllabus: Unit 3 covers Arrays, Linked Lists, Stacks, Queues, Trees, Graphs""")
]).partial(format_instructions=architect_parser.get_format_instructions())

architect_chain = architect_prompt | llm | architect_parser

def run_architect(current_unit: int, current_topic: int, mastery_level: float):
    result = architect_chain.invoke({
        "current_unit": current_unit,
        "current_topic": current_topic,
        "mastery_level": mastery_level
    })
    
    return {
        "next_topic": result.next_topic,
        "hours": result.hours,
        "remedial_needed": result.remedial_needed,
        "reason": result.reason
    }

# ============================================
# TEST SECTION
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("TESTING ARCHITECT AGENT (FREE - Groq)")
    print("=" * 50)
    
    print("\n--- Test 1: Student scored 45% ---")
    plan1 = run_architect(current_unit=3, current_topic=2, mastery_level=0.45)
    print(f"Next Topic: {plan1['next_topic']}")
    print(f"Hours: {plan1['hours']}")
    print(f"Remedial: {plan1['remedial_needed']}")
    print(f"Reason: {plan1['reason']}")
    
    print("\n--- Test 2: Student scored 80% ---")
    plan2 = run_architect(current_unit=3, current_topic=2, mastery_level=0.80)
    print(f"Next Topic: {plan2['next_topic']}")
    print(f"Hours: {plan2['hours']}")
    print(f"Remedial: {plan2['remedial_needed']}")
    print(f"Reason: {plan2['reason']}")