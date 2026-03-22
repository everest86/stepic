import csv
import re

INPUT_FILE = "export.csv"
OUTPUT_FILE = "parsed.csv"

# регулярка для ID прошивки (можно расширить)
ID_PATTERNS = [
    r'\b\d{8,12}\b',  # длинные числовые ID (Bosch и т.д.)
    r'\b[A-Z0-9]{8,}\b',  # общий формат
]


def extract_id(filename):
    name = re.sub(r'\.[^.]+$', '', filename)  # убираем расширение

    for pattern in ID_PATTERNS:
        match = re.search(pattern, name)
        if match:
            return match.group(0)

    return None


def is_valid_ecu(text):
    # типичные слова для ЭБУ
    keywords = [
        "EDC", "ME", "MED", "SID", "Delphi",
        "Bosch", "Denso", "Siemens", "Valeo"
    ]
    return any(k.lower() in text.lower() for k in keywords)


def parse_row(row):
    try:
        filename = row[1]
        name = row[2]
        full_path = row[3]

        parts = [p.strip() for p in full_path.split(">")]

        # минимум: Марка > Модель > ЭБУ
        if len(parts) < 3:
            return None

        brand = parts[0]
        model = parts[1]
        ecu = parts[2]

        # фильтр на валидность ЭБУ
        if not is_valid_ecu(ecu):
            return None

        file_id = extract_id(name)
        if not file_id:
            return None

        return {
            "brand": brand,
            "model": model,
            "ecu": ecu,
            "id": file_id,
            "filename": filename
        }

    except Exception:
        return None


def main():
    results = []

    with open(INPUT_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            parsed = parse_row(row)
            if parsed:
                results.append(parsed)

    # сохраняем
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["brand", "model", "ecu", "id", "filename"])

        for r in results:
            writer.writerow([
                r["brand"],
                r["model"],
                r["ecu"],
                r["id"],
                r["filename"]
            ])

    print(f"Готово! Найдено {len(results)} валидных записей.")


if __name__ == "__main__":
    main()