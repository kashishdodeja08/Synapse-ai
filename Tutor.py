# Tutor.py
# Agent 2: The Teacher - Creates exam-oriented lessons using textbook content

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ============================================
# STEP 1: Load API key (same as Agents.py)
# ============================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
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
# STEP 3: Define what Tutor outputs
# ============================================

class TutorLesson(BaseModel):
    title: str = Field(description="Lesson title")
    content: str = Field(description="Detailed lesson with LaTeX math and code")
    key_points: list[str] = Field(description="3-5 key takeaways for exams")
    practice_questions: list[str] = Field(description="2 practice problems")
    time_complexity_notes: str = Field(description="Time complexity analysis")

tutor_parser = PydanticOutputParser(pydantic_object=TutorLesson)

# ============================================
# STEP 4: Create prompt for Tutor
# ============================================

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

# ============================================
# STEP 5: Build chain and function
# ============================================

tutor_chain = tutor_prompt | llm | tutor_parser

def run_tutor(topic: str, hours: int, remedial: bool, textbook_context: str):
    """
    Runs the Tutor agent.
    
    Input: topic, hours, remedial flag, textbook content
    Output: structured lesson
    """
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
# TEST SECTION
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING TUTOR AGENT")
    print("=" * 50)
    
    # Fake textbook content (replace with ChromaDB later)
    fake_textbook = """
    Stacks: A stack is a linear data structure that follows LIFO (Last In First Out).
    Operations: push (add), pop (remove), peek (view top).
    Time complexity: All operations are O(1).
    Applications: Function call stack, undo operations, expression evaluation.
    
    Queues: A queue follows FIFO (First In First Out).
    Operations: enqueue (add rear), dequeue (remove front), front (view).
    Time complexity: All operations are O(1).
    Applications: CPU scheduling, printer queue, BFS algorithm.
    """
    
    print("\n--- Test: Teaching Stacks (Remedial) ---")
    lesson = run_tutor(
        topic="Stacks and Queues",
        hours=4,
        remedial=True,
        textbook_context=fake_textbook
    )
    
    print(f"\nTitle: {lesson['title']}")
    print(f"\nContent preview:\n{lesson['content'][:300]}...")
    print(f"\nKey Points: {lesson['key_points']}")
    print(f"\nPractice Questions: {lesson['practice_questions']}")
    print(f"\nTime Complexity: {lesson['time_complexity_notes']}")