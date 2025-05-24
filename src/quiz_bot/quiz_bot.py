import os.path
import re
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("OPEN_AI_ENDPOINT"),
    api_key=os.getenv("API_OPEN_AI_KEY"),
)
MODEL_NAME = "gpt-4o"
DEPLOYMENT = "gpt-4o"

chat_prompt_overall_14 = [
    {
        "role": "system",
        "content": """Jesteś quiz botem, który zadaje bardzo ciekawe, kreatywne i angażujące pytania 
        wielokrotnego wyboru (tylko jedna poprawna odpowiedź). Twoim zadaniem jest zadawać pytania z 
        różnych dziedzin wiedzy  w języku polskim.
        Pytania mają być nietuzinkowe, intrygujące i zachęcać do myślenia. 
        Każde pytanie ma mieć dokładnie cztery opcje odpowiedzi oznaczone A, B, C, D, z jedną poprawną 
        odpowiedzią.
        Zasady formatowania:
        Pytanie rozpoczynaj nową linijką.
        Odpowiedzi zapisuj w kolejnych linijkach, każda z literą A-D i spacją.
        Na końcu zawsze podawaj poprawną odpowiedź w formacie: "Odpowiedź: X" 
        (gdzie X to litera A, B, C lub D)."""},
    
]
chat_prompt_overall_57 = [
    {
        "role": "system",
        "content": """Jesteś quiz botem, który zadaje bardzo ciekawe, kreatywne i angażujące pytania 
        wielokrotnego wyboru (tylko jedna poprawna odpowiedź). Twoim zadaniem jest w języku polskim.
        Pytania mają być nietuzinkowe, intrygujące i zachęcać do myślenia. 
        Każde pytanie ma mieć dokładnie 3 opcje odpowiedzi oznaczone A, B, C z jedną poprawną 
        odpowiedzią.
        Zasady formatowania:
        Pytanie rozpoczynaj nową linijką.
        Odpowiedzi zapisuj w kolejnych linijkach, każda z literą A-C i spacją.
        Na końcu zawsze podawaj poprawną odpowiedź w formacie: "Odpowiedź: X" 
        (gdzie X to litera A, B, C).
            """},
    
]

chat_prompt_overall_8 = [{"role": "system",
        "content": """Jesteś quiz botem, który zadaje bardzo ciekawe, kreatywne i angażujące pytania 
        wielokrotnego wyboru (tylko jedna poprawna odpowiedź). Twoim zadaniem jest w języku polskim.
        Pytania mają być nietuzinkowe, intrygujące i zachęcać do myślenia. 
        Każde pytanie ma mieć dokładnie 2 opcje odpowiedzi oznaczone A, B z jedną poprawną 
        odpowiedzią.
        Zasady formatowania:
        Pytanie rozpoczynaj nową linijką.
        Odpowiedzi zapisuj w kolejnych linijkach, każda z literą A-B i spacją.
        Na końcu zawsze podawaj poprawną odpowiedź w formacie: "Odpowiedź: X" 
        (gdzie X to litera A, B).
            """}]



def choose_topic():
    response = client.chat.completions.create(
        messages=[
            {
                "role": "assistant",
                "content": """Wygeneruj 2 losowe tematyki pytań do gry postaw na milion w formacie:
                Tematyka 1: [tematyka 1]
                Tematyka 2: [tematyka 2]"""
            }
        ],
        max_tokens=768,
        temperature=1.0,
        top_p=1.0,
        model=DEPLOYMENT
    )
    
    
    content = response.choices[0].message.content

    # Wydobywanie tematyki z tekstu
    matches = re.findall(r"Tematyka \d+: (.+)", content)
    
    return matches  

def update_history(question):
    chat_prompt_overall_14.append(question)
    chat_prompt_overall_57.append(question)
    chat_prompt_overall_8.append(question)
def get_question(topic,question_num):
    prompt = f"Wygeneruj pytanie z tematyki {topic} w języku polskim. Unikaj pytań, które już były."
    if question_num <= 4:
        prompt = f"Wygeneruj pytanie z tematyki {topic} w języku polskim. Unikaj pytań, które już były."
        chat_prompt_overall_14.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=chat_prompt_overall_14,
        temperature=1.0,
        max_tokens=350
        )

        content = response.choices[0].message.content
        update_history(content)

        match = re.search(r'Odpowiedź: ([A-D])', content)
        if not match:
            raise ValueError("Nie znaleziono poprawnej odpowiedzi w propmtcie.")

        answer = match.group(1)
        question_text = re.sub(r'Odpowiedź: ([A-D])', '', content).strip()
        return question_text, answer
    elif question_num <= 7:
        prompt = f"Wygeneruj pytanie z tematyki {topic} w języku polskim. Unikaj pytań, które już były."
        chat_prompt_overall_57.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=chat_prompt_overall_57,
        temperature=1.0,
        max_tokens=350
        )
        content = response.choices[0].message.content
        update_history(content)
        match = re.search(r'Odpowiedź: ([A-C])', content)
        if not match:
            raise ValueError("Nie znaleziono poprawnej odpowiedzi w propmtcie.")

        answer = match.group(1)
        question_text = re.sub(r'Odpowiedź: ([A-C])', '', content).strip()
        return question_text, answer
    else:
        prompt = f"Wygeneruj pytanie z tematyki {topic} w języku polskim. Unikaj pytań, które już były."
        chat_prompt_overall_8.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=chat_prompt_overall_8,
        temperature=1.0,
        max_tokens=350
        )
        content = response.choices[0].message.content
        update_history(content)
        match = re.search(r'Odpowiedź: ([A-B])', content)
        if not match:
            raise ValueError("Nie znaleziono poprawnej odpowiedzi w propmtcie.")

        answer = match.group(1)
        question_text = re.sub(r'Odpowiedź: ([A-B])', '', content).strip()
        return question_text, answer

def save_log(question_num, question_text, correct_answer, bets, prize):
    log_path = os.path.abspath("prompts/best.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Pytanie {question_num}:\n")
        f.write(question_text + "\n")
        f.write(f"Poprawna odpowiedź: {correct_answer}\n")
        f.write(f"Obstawienia: {bets}\n")
        f.write(f"Stan konta po pytaniu: {prize} zł\n")
        f.write("-" * 40 + "\n")

def place_bet(possible_answers, correct_answer, current_money):
    print("\nMożesz postawić pieniądze na jedną lub więcej odpowiedzi.")
    print("Możesz rozłożyć pieniądze na wszystkie odpowiedzi z wyjątkiem jednej.")
    print("Pamiętaj, że suma nie może przekroczyć posiadanych pieniędzy.\n")

    bets = {}
    total_bet = 0

    for ans in possible_answers:
        while True:
            try:
                bet = input(f"Ile chcesz postawić na odpowiedź {ans}? (wpisz 0, jeśli nic): ")
                bet = int(bet)
                if bet < 0:
                    print("Nie możesz postawić ujemnej kwoty.")
                else:
                    bets[ans] = bet
                    total_bet += bet
                    break
            except ValueError:
                print("Nieprawidłowa wartość. Podaj liczbę całkowitą.")

    if total_bet > current_money:
        print(f"Przekroczono dostępne środki! Masz tylko {current_money} zł.")
        return place_bet(possible_answers, correct_answer, current_money), bets

    if len([b for b in bets.values() if b > 0]) > len(possible_answers) - 1:
        print("Możesz obstawić maksymalnie tyle odpowiedzi, ile jest możliwych minus jedna.")
        return place_bet(possible_answers, correct_answer, current_money), bets

    if bets.get(correct_answer, 0) > 0:
        print(f"\nPoprawna odpowiedź to: {correct_answer}. Gratulacje! Przechodzisz dalej.")
        return bets[correct_answer], bets  # Kwota, która przechodzi dalej, oraz obstawienia
    else:
        print(f"\nPoprawna odpowiedź to: {correct_answer}. Niestety, przegrałeś wszystko.")
        return 0, bets

def main():
    prize = 1000000
    question = 1
    print("Witaj w grze postaw na milion!")
    print("Zasady są proste: odpowiadaj na pytania i zdobywaj pieniądze!")
    print("Na pytania od 1-4 masz 4 odpowiedzi do wyboru, na pytania od 5-7 masz 3 odpowiedzi do wyboru, a na pytanie od 8 masz 2 odpowiedzi do wyboru.")
    print("Mozesz polozyc pieniadze na jedna odpowiedz lub na wszystkie oprocz jednej odpowiedzi, ale tak aby "
          "twoja dotychczasowa suma pieniedzy nie przekraczala limitu.")
    print("Powodzenie w grze!")

    while question <= 8 and prize > 0:
        print(f"\nPytanie {question}:")
        print(f"Na szali masz {prize} zł")
        print("Wybierz tematykę pytania:")
        topics = choose_topic()
        print(f"Tematyka 1: {topics[0]}")
        print(f"Tematyka 2: {topics[1]}")
        print("Wybierz tematykę pytania (1 lub 2):")
        while True:
            try:
                choice = int(input("Wpisz 1 lub 2, aby wybrać tematykę: "))
                if choice in [1, 2]:
                    break
                else:
                    print("Niepoprawny wybór. Wpisz 1 lub 2.")
            except ValueError:
                print("To nie jest liczba. Spróbuj jeszcze raz.")

        selected_topic = topics[choice - 1]
        print(f"Wybrałeś tematykę: {selected_topic}")
        quest, ans = get_question(selected_topic, question)
        print(f"\nPytanie: {quest}")
        if question <= 4:
            options = ['A', 'B', 'C', 'D']
        elif question <= 7:
            options = ['A', 'B', 'C']
        else:
            options = ['A', 'B']

        new_prize, bets = place_bet(options, ans, prize)
        save_log(question, quest, ans, bets, new_prize)
        prize = new_prize
        question += 1

    print(f"\nKoniec gry! Twój końcowy stan konta to: {prize} zł")
    print("Dziękujemy za grę!")


if __name__ == "__main__":
    main()






