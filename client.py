import requests
import os

BASE_URL = "http://127.0.0.1:8000"

def list_pdfs():
    r = requests.get(f"{BASE_URL}/list_pdfs")
    print("Indexed PDFs:", r.json()["pdfs"])

def upload_pdf():
    path = input("Enter path to PDF: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return
    with open(path, "rb") as f:
        r = requests.post(f"{BASE_URL}/upload_pdf", files={"file": f})
    print(r.json())

def delete_pdf():
    pdf_name = input("Enter PDF name to delete (without .pdf): ").strip()
    r = requests.delete(f"{BASE_URL}/delete_pdf/{pdf_name}")
    print(r.json())

def reset_db():
    confirm = input("Are you sure you want to reset everything? (yes/no): ")
    if confirm.lower() == "yes":
        r = requests.post(f"{BASE_URL}/reset")
        print(r.json())
    else:
        print("Reset cancelled.")

def search_pdf():
    query = input("Enter your search query: ").strip()
    pdf_choice = input("Search in one PDF (enter name) or press Enter for all: ").strip()
    params = {"query": query, "top_k": 3}
    if pdf_choice:
        params["pdf"] = pdf_choice
    r = requests.get(f"{BASE_URL}/search", params=params)
    data = r.json()
    print("\nSearch Results:")
    for res in data.get("results", []):
        print(f"- PDF: {res['pdf']} | Page: {res['page']} | Score: {res['score']:.4f}")
        print(f"  Text: {res['text']}\n")

def main():
    while True:
        print("\n=== PDF Vector DB Client ===")
        print("1. List PDFs")
        print("2. Upload PDF")
        print("3. Delete PDF")
        print("4. Reset DB")
        print("5. Search")
        print("0. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            list_pdfs()
        elif choice == "2":
            upload_pdf()
        elif choice == "3":
            delete_pdf()
        elif choice == "4":
            reset_db()
        elif choice == "5":
            search_pdf()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()

