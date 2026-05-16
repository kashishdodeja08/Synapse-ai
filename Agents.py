# Agents.py
# All 4 agents in one file
# Day 4: All agents + manual chain test

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

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
# AGENT 3: EVALUATOR (The Examiner)
# ============================================

class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly 4 options A, B, C, D")
    correct_index: int = Field(description="0=A, 1=B, 2=C, 3=D")
    explanation: str = Field(description="Why correct answer is right")

class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(description="5 multiple choice questions")
    coding_problems: List[str] = Field(description="2 coding problems with descriptions")
    topic: str = Field(description="Topic being tested")

quiz_parser = PydanticOutputParser(pydantic_object=Quiz)

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict MLSU Examiner. Create challenging but fair assessments.
    
    {format_instructions}
    
    RULES:
    - Generate EXACTLY 5 MCQs based ONLY on the lesson content provided
    - Generate EXACTLY 2 coding problems
    - Each MCQ must have 4 options with exactly one correct answer
    - Questions must test understanding, not just memorization
    - Include common misconceptions as distractors (wrong options)
    - Explanations must cite specific content from the lesson
    - Coding problems must be solvable using concepts from the lesson"""),
    
    ("human", """Create a quiz based on this lesson:
    
    {lesson_content}
    
    Topic: {topic}
    Is remedial: {remedial}""")
]).partial(format_instructions=quiz_parser.get_format_instructions())

quiz_chain = quiz_prompt | llm | quiz_parser

class QuizResult(BaseModel):
    score_percent: int = Field(description="Score 0-100")
    passed: bool = Field(description="True if score >= 60")
    weak_areas: List[str] = Field(description="Specific subtopics failed")
    feedback: str = Field(description="Detailed feedback for student")

result_parser = PydanticOutputParser(pydantic_object=QuizResult)

def run_evaluator(lesson_content: str, topic: str, remedial: bool, simulated_answers: dict = None):
    """
    Evaluator generates quiz and grades it.
    For now, we simulate student answers.
    In real version, Frontend sends answers and we grade.
    """
    
    # Generate quiz using AI
    quiz = quiz_chain.invoke({
        "lesson_content": lesson_content[:1500],
        "topic": topic,
        "remedial": remedial
    })
    
    # SIMULATE GRADING
    if simulated_answers is None:
        simulated_score = 45
    else:
        simulated_score = simulated_answers.get("score", 45)
    
    passed = simulated_score >= 60
    
    if not passed:
        weak_areas = ["Core concepts", "Application problems"]
    else:
        weak_areas = []
    
    return {
        "quiz": {
            "questions": [q.model_dump() for q in quiz.questions],
            "coding_problems": quiz.coding_problems,
            "topic": quiz.topic
        },
        "result": {
            "score_percent": simulated_score,
            "passed": passed,
            "weak_areas": weak_areas,
            "feedback": f"Score: {simulated_score}%. {'Passed' if passed else 'Remedial session needed.'}"
        },
        "mastery_level": simulated_score / 100
    }

# ============================================
# AGENT 4: RESEARCHER (The Fact-Checker)
# ============================================

class Verification(BaseModel):
    status: str = Field(description="VERIFIED or ERROR")
    issues: List[str] = Field(description="List of incorrect claims found")
    corrections: str = Field(description="Corrected explanation if errors found")
    sources: List[str] = Field(description="Official sources checked")

verify_parser = PydanticOutputParser(pydantic_object=Verification)

verify_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous Research Librarian. You verify technical claims.
    
    {format_instructions}
    
    RULES:
    - Check ALL time complexity claims (O(n), O(log n), O(n²), etc.)
    - Verify algorithm steps and properties
    - Confirm code syntax is correct
    - Check definitions against standard computer science references
    - Be STRICT: a small error is still an error
    - If everything is correct, say VERIFIED with empty issues list
    - If errors found, list each issue with specific correction"""),
    
    ("human", """Verify these technical claims from a lesson:
    
    {lesson_content}
    
    Check:
    1. Are all time complexities correct?
    2. Are algorithm steps accurate?
    3. Are definitions correct?
    4. Is the code syntactically valid?""")
]).partial(format_instructions=verify_parser.get_format_instructions())

verify_chain = verify_prompt | llm | verify_parser

def run_researcher(lesson_content: str):
    """
    Researcher verifies tutor content.
    For now, uses AI self-check (no real web search yet).
    """
    result = verify_chain.invoke({
        "lesson_content": lesson_content[:1500]
    })
    
    return {
        "status": result.status,
        "issues": result.issues,
        "corrections": result.corrections,
        "sources": result.sources
    }

# ============================================
# TEST SECTION - MUST BE LAST
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ALL 4 AGENTS (MANUAL CHAIN)")
    print("=" * 60)
    
    # Step 1: Architect plans
    print("\n--- STEP 1: ARCHITECT ---")
    plan = run_architect(current_unit=3, current_topic=2, mastery_level=0.45)
    print(f"Plan: {plan['next_topic']}, {plan['hours']}h, Remedial: {plan['remedial_needed']}")
    
    # Step 2: Tutor teaches
    print("\n--- STEP 2: TUTOR ---")
    fake_textbook = "Stacks follow LIFO. Push and pop are O(1). Applications: function calls, undo."
    lesson = run_tutor(
        topic=plan['next_topic'],
        hours=plan['hours'],
        remedial=plan['remedial_needed'],
        textbook_context=fake_textbook
    )
    print(f"Lesson: {lesson['title']}")
    
    # Step 3: Evaluator quizzes
    print("\n--- STEP 3: EVALUATOR ---")
    eval_result = run_evaluator(
        lesson_content=lesson['content'],
        topic=plan['next_topic'],
        remedial=plan['remedial_needed'],
        simulated_answers={"score": 45}
    )
    print(f"Score: {eval_result['result']['score_percent']}%")
    print(f"Passed: {eval_result['result']['passed']}")
    
    # Step 4: Researcher verifies
    print("\n--- STEP 4: RESEARCHER ---")
    verify = run_researcher(lesson_content=lesson['content'])
    print(f"Status: {verify['status']}")
    if verify['issues']:
        print(f"Issues found: {verify['issues']}")
    else:
        print("No issues found")
    
    print("\n" + "=" * 60)
    print("NEXT: LangGraph automatic loop")
    print("=" * 60)