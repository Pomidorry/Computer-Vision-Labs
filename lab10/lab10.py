import time
import json
import numpy as np
from pynput import keyboard
from scipy import stats
import os
import datetime

def remove_outliers(data, m=3):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    filtered_data = data[abs(data - mean) < m * std]
    return filtered_data.tolist()

def save_biometric_parameters(file_path, mean_val, var_val):
    reference_data = {
        "mean": mean_val,
        "variance": var_val
    }
    with open(file_path, 'w') as f:
        json.dump(reference_data, f, indent=4)

def load_biometric_parameters(file_path):
    if not os.path.exists(file_path):
        return None, None
    with open(file_path, 'r') as f:
        ref_data = json.load(f)
        return ref_data['mean'], ref_data['variance']

def collect_intervals(control_phrase, message=""):
    if not message:
        message = f"Наберіть контрольну фразу «{control_phrase}»:"
    print(message)

    intervals = []
    input_times = []

    def on_press(key):
        input_times.append(time.time())
        if len(input_times) >= len(control_phrase):
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Обчислюємо часові інтервали
    for i in range(1, len(input_times)):
        intervals.append(input_times[i] - input_times[i-1])

    return intervals

def compare_statistics_classical(ref_mean, ref_var, test_mean, test_var, alpha=0.05):
    n_ref = 30
    n_test = 30

    numerator = ref_mean - test_mean
    denominator = np.sqrt(ref_var/n_ref + test_var/n_test)
    T_calc = numerator/denominator if denominator != 0 else 0

    var_part = (ref_var / n_ref) + (test_var / n_test)
    numerator_df = var_part**2
    denominator_df = ((ref_var**2)/(n_ref**2*(n_ref-1))) + ((test_var**2)/(n_test**2*(n_test-1)))
    df_t = numerator_df / denominator_df if denominator_df != 0 else (n_ref + n_test - 2)

    T_crit = stats.t.ppf(1 - alpha/2, df_t if df_t > 0 else 1)
    reject_mean = (abs(T_calc) > T_crit)

    if ref_var >= test_var:
        F_calc = ref_var / (test_var if test_var!=0 else 1e-8)
        df1, df2 = n_ref-1, n_test-1
    else:
        F_calc = test_var / (ref_var if ref_var!=0 else 1e-8)
        df1, df2 = n_test-1, n_ref-1

    F_crit_upper = stats.f.ppf(1 - alpha/2, df1 if df1>0 else 1, df2 if df2>0 else 1)
    F_crit_lower = stats.f.ppf(alpha/2, df1 if df1>0 else 1, df2 if df2>0 else 1)
    reject_var = (F_calc > F_crit_upper) or (F_calc < F_crit_lower)

    return (not reject_mean) and (not reject_var)

def init_training_log(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("timestamp;attempt;count;mean;var\n")

def init_identification_log(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("timestamp;attempt;test_mean;test_var;alpha;verdict\n")

import datetime

def append_to_training_log(filename, attempt_num, intervals, mean_val, var_val):
    """
    Записує проміжний результат тренування (дописування, режим 'a').
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp};{attempt_num};{len(intervals)};{mean_val:.5f};{var_val:.5f}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line)

def append_to_identification_log(filename, attempt_num, test_mean, test_var, alpha, identified):
    """
    Записує проміжний результат ідентифікації (дописування, режим 'a').
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "IDENTIFIED" if identified else "NOT_IDENTIFIED"
    line = (f"{timestamp};{attempt_num};{test_mean:.5f};"
            f"{test_var:.5f};{alpha};{verdict}\n")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line)

def training_flow(reference_file, training_log_file):
    """
    Етап навчання за блок-схемою.
    Лог-файл training_log_file перезаписується при кожному виклику.
    """
    print("\n=== РЕЖИМ НАВЧАННЯ ===")
    try:
        num_attempts = int(input("Введіть кількість спроб навчання: "))
    except ValueError:
        num_attempts = 3
        print("Некоректний ввід, кількість спроб = 3 за замовчуванням.")

    control_phrase = input("Введіть контрольну фразу: ")
    if not control_phrase.strip():
        print("Порожня фраза! Вихід.")
        return

    init_training_log(training_log_file)

    all_intervals = []
    current_attempt = 0

    while current_attempt < num_attempts:
        current_attempt += 1
        print(f"\n--- Спроба навчання № {current_attempt} / {num_attempts}")
        intervals = collect_intervals(control_phrase, message="Введіть фразу для навчання:")
        
        intervals = remove_outliers(intervals)

        if len(intervals) > 0:
            mean_val = float(np.mean(intervals))
            var_val = float(np.var(intervals, ddof=1))
        else:
            mean_val = 0.0
            var_val  = 0.0

        append_to_training_log(training_log_file, current_attempt, intervals, mean_val, var_val)

        all_intervals.extend(intervals)

    all_intervals = remove_outliers(all_intervals)
    if len(all_intervals) < 1:
        print("Недостатньо даних для формування еталону!")
        return

    final_mean = float(np.mean(all_intervals))
    final_var  = float(np.var(all_intervals, ddof=1))

    save_biometric_parameters(reference_file, final_mean, final_var)
    print(f"\n[НАВЧАННЯ] Еталонні характеристики збережено у файл: {reference_file}")
    print(f"mean = {final_mean:.5f}, var = {final_var:.5f}")


def identification_flow(reference_file, identification_log_file):
    """
    Етап ідентифікації за блок-схемою.
    Лог-файл identification_log_file перезаписується при кожному виклику.
    Якщо хоча б половина спроб дає 'ідентифіковано' => користувач ідентифікований.
    """
    print("\n=== РЕЖИМ ІДЕНТИФІКАЦІЇ ===")

    try:
        num_attempts = int(input("Введіть кількість спроб ідентифікації: "))
    except ValueError:
        num_attempts = 2
        print("Некоректний ввід, кількість спроб = 2 за замовчуванням.")

    try:
        alpha = float(input("Введіть рівень значущості (наприклад, 0.05): "))
    except ValueError:
        alpha = 0.05
        print("Некоректний ввід, рівень значущості = 0.05 за замовчуванням.")

    control_phrase = input("Введіть контрольну фразу: ")
    if not control_phrase.strip():
        print("Порожня фраза! Вихід.")
        return

    init_identification_log(identification_log_file)

    ref_mean, ref_var = load_biometric_parameters(reference_file)
    if ref_mean is None or ref_var is None:
        print("Немає еталонних характеристик! Спочатку виконайте навчання.")
        return

    success_count = 0

    for attempt_num in range(1, num_attempts+1):
        print(f"\n--- Спроба ідентифікації №{attempt_num} / {num_attempts}")
        intervals = collect_intervals(control_phrase, message="Введіть фразу для ідентифікації:")
        intervals = remove_outliers(intervals)

        if len(intervals) > 0:
            test_mean = float(np.mean(intervals))
            test_var  = float(np.var(intervals, ddof=1))
        else:
            test_mean = 0.0
            test_var  = 0.0

        # Порівняння
        result = compare_statistics_classical(ref_mean, ref_var, test_mean, test_var, alpha)
        
        append_to_identification_log(
            identification_log_file, 
            attempt_num, 
            test_mean, 
            test_var, 
            alpha, 
            identified=result
        )

        if result:
            success_count += 1

    # Фінальний вердикт
    print("\n===== ПІДСУМКОВИЙ ВЕРДИКТ ЗА УСІМА СПРОБАМИ =====")
    print(f"Успішних спроб: {success_count} з {num_attempts}")
    if success_count >= (num_attempts / 2):
        print("Користувача можна вважати ІДЕНТИФІКОВАНИМ (>= 50% успішних спроб).")
    else:
        print("Користувач НЕ ідентифікований (менше половини успішних спроб).")


if __name__ == "__main__":
    # Файл, де зберігатимуться еталонні параметри
    reference_file = "C:\labsCV\lab10\\biometric_reference.json"
    # Лог-файли
    training_log_file = "C:\labsCV\lab10\\training_log.csv"
    identification_log_file = "C:\labsCV\lab10\identification_log.csv"

    while True:
        print("\n===============================")
        print("  1) Навчання")
        print("  2) Ідентифікація")
        print("  0) Вихід")
        choice = input("Оберіть режим: ")

        if choice == "1":
            training_flow(reference_file, training_log_file)
        elif choice == "2":
            identification_flow(reference_file, identification_log_file)
        elif choice == "0":
            print("Роботу завершено.")
            break
        else:
            print("Невідома команда, спробуйте ще раз.")
