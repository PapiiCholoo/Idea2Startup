
def run_quiz():
    print("Welcome to the Quiz App!\n")
    print("Answer the following questions:\n")

    questions = [
        {"q": "What is the capital of France?", "a": "paris"},
        {"q": "Which planet is known as the Red Planet?", "a": "mars"},
        {"q": "What is 5 + 7?", "a": "12"},
    ]

    score = 0
    for i, item in enumerate(questions, 1):
        user_answer = input(f"Q{i}: {item['q']} ").strip().lower()
        if user_answer == item["a"]:
            print("✅ Correct!\n")
            score += 1
        else:
            print(f"❌ Wrong. The correct answer is {item['a'].title()}.\n")

    print(f"Your final score: {score}/{len(questions)}")
    if score == len(questions):
        print("🎉 Excellent work!")
    elif score >= len(questions)//2:
        print("👍 Good job! Keep practicing.")
    else:
        print("📘 Don’t worry, review and try again!")

if __name__ == "__main__":
    run_quiz()
