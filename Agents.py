# Agents.py
# All 4 agents in one file
# Day 3: Architect + Tutor connected

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ============================================
# SHARED: API key, LLM setup
# ============================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    exit(1)

print(f"Key loaded: {api_key[:15]}...")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)

# ============================================
# AGENT 1: ARCHITECT (Planner)
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
# AGENT 2: TUTOR (Teacher)
# ============================================

class TutorLesson(BaseModel):
    title: str = Field(description="Lesson title")
    content: str = Field(description="Detailed lesson with LaTeX math and code")
    key_points: list[str] = Field(description="3-5 key takeaways for exams")
    practice_questions: list[str] = Field(description="2 practice problems")
    time_complexity_notes: str = Field(description="Time complexity analysis")

tutor_parser = PydanticOutputParser(pydantic_object=TutorLesson)

tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Technical Tutor for BCA students at MLSU.
    You teach exam-oriented content with mathematical rigor.
    
    {format_instructions}
    
    RULES (follow strictly):
    - Use LaTeX for ALL math: $O(n^2)$, $T(n) = 2T(n/2) + O(n)$
    - NO analogies (never say "like a stack of plates")
    - Use technical definitions only
    - Include code examples in ```python blocks
    - Focus on what appears in MLSU exams
    - Explain WHY, not just WHAT"""),
    
    ("human", """Teach this topic: {topic}
    
    Hours allocated: {hours}
    Is remedial session: {remedial}
    
    Textbook context to use (your ONLY source):
    {textbook_context}
    
    Write a complete lesson.""")
]).partial(format_instructions=tutor_parser.get_format_instructions())

tutor_chain = tutor_prompt | llm | tutor_parser

def run_tutor(topic: str, hours: int, remedial: bool, textbook_context: str):
    result = tutor_chain.invoke({
        "topic": topic,
        "hours": hours,
        "remedial": remedial,
        "textbook_context": textbook_context
    })
    
    return {
        "title": result.title,
        "content": result.content,
        "key_points": result.key_points,
        "practice_questions": result.practice_questions,
        "time_complexity_notes": result.time_complexity_notes
    }

# ============================================
# TEST: Architect plans → Tutor teaches
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING: ARCHITECT → TUTOR FLOW")
    print("=" * 60)
    
    # Fake textbook (replace with ChromaDB on Day 7)
    fake_textbook = """
    Stacks: A stack is a linear data structure that follows LIFO (Last In First Out).
    Operations: push (add), pop (remove), peek (view top).
    Time complexity: All operations are O(1).
    Applications: Function call stack, undo operations, expression evaluation.
    """
    
    # STEP 1: Architect plans
    print("\n--- STEP 1: ARCHITECT PLANS ---")
    plan = run_architect(current_unit=3, current_topic=2, mastery_level=0.45)
    print(f"Next Topic: {plan['next_topic']}")
    print(f"Hours: {plan['hours']}")
    print(f"Remedial: {plan['remedial_needed']}")
    print(f"Reason: {plan['reason']}")
    
    # STEP 2: Tutor teaches (uses Architect's plan)
    print("\n--- STEP 2: TUTOR TEACHES ---")
    lesson = run_tutor(
        topic=plan['next_topic'],
        hours=plan['hours'],
        remedial=plan['remedial_needed'],
        textbook_context=fake_textbook
    )
    
    print(f"\nTitle: {lesson['title']}")
    print(f"\nContent preview:\n{lesson['content'][:250]}...")
    print(f"\nKey Points: {lesson['key_points']}")
    print(f"\nPractice Questions: {lesson['practice_questions']}")