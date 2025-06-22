import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
from .cleaner import clean_data
from .classify import classify_cleaned
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def launch_gui():
    df = None
    X, y = None, None
    chart_canvas = None

    def browse_file():
        nonlocal df
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                df = pd.read_csv(filepath)
                file_entry.delete(0, tk.END)
                file_entry.insert(0, filepath)

                # Update target dropdown values
                cols = list(df.columns)
                target_dropdown['values'] = cols
                if cols:
                    target_var.set(cols[-1])  # default to last col
                
                messagebox.showinfo("File Loaded", f"Loaded {filepath}\nSelect target column.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")

    def run_cleaning():
        nonlocal X, y
        if df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        target = target_var.get()
        if not target:
            messagebox.showerror("Error", "Please select a target column.")
            return
        
        try:
            X, y = clean_data(df, target)
            results_box.delete("1.0", tk.END)
            results_box.insert(tk.END, "✅ Data Cleaning Complete!\n\n")
            results_box.insert(tk.END, f"Features shape: {X.shape}\nTarget shape: {y.shape}\n\n")
            results_box.insert(tk.END, f"Features (first 5 rows):\n{X.head().to_string(index=False)}\n\n")
            results_box.insert(tk.END, f"Target (first 5):\n{y.head().to_string(index=False)}\n")
            results_box.yview_moveto(0.0)
        except Exception as e:
            messagebox.showerror("Cleaning Error", str(e))

    def run_classification():
        nonlocal chart_canvas
        if X is None or y is None:
            messagebox.showerror("Error", "Please clean the data before classification.")
            return

        try:
            results, bestmodel = classify_cleaned(X, y, verbose=False)

            results_box.insert(tk.END, "\n\n🔍 Classification Results:\n\n")
            for model, metrics in results.items():
                results_box.insert(tk.END, f"🔹 {model}\n")
                for metric, value in metrics.items():
                    results_box.insert(tk.END, f"   {metric.title()}: {value:.2f}\n")
                results_box.insert(tk.END, "\n")

            if bestmodel:
                best_acc = results[bestmodel]["accuracy"]
                results_box.insert(tk.END, f"🏆 Best Model: {bestmodel} with Accuracy: {best_acc:.2f}\n")
            results_box.yview_moveto(0.0)

            # Plot Accuracy Bar Chart
            accuracies = [metrics["accuracy"] for metrics in results.values()]
            model_names = list(results.keys())

            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            bars = ax.bar(model_names, accuracies, color=["#4CAF50", "#2196F3", "#FFC107"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Accuracy Comparison")
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{acc:.2f}", ha='center', fontsize=10)

            if chart_canvas:
                chart_canvas.get_tk_widget().destroy()

            chart_canvas = FigureCanvasTkAgg(fig, master=root)
            chart_canvas.draw()
            chart_canvas.get_tk_widget().grid(row=5, column=0, columnspan=3, pady=10)

        except Exception as e:
            messagebox.showerror("Classification Error", str(e))


    # ======== GUI Layout ========
    root = tk.Tk()
    root.title("🧹 CleanClass: Clean + Classify + Visualize")
    root.geometry("900x750")
    root.configure(bg="#f0f0f5")

    font_label = ("Segoe UI", 11)
    font_btn = ("Segoe UI", 11, "bold")
    font_entry = ("Segoe UI", 10)
    font_results = ("Consolas", 10)

    # File select row
    tk.Label(root, text="CSV File:", bg="#f0f0f5", font=font_label).grid(row=0, column=0, sticky="e", padx=5, pady=5)
    file_entry = tk.Entry(root, width=60, font=font_entry)
    file_entry.grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_file, bg="#1976D2", fg="white", font=font_btn).grid(row=0, column=2, padx=5, pady=5)

    # Target column dropdown
    tk.Label(root, text="Target Column:", bg="#f0f0f5", font=font_label).grid(row=1, column=0, sticky="e", padx=5, pady=5)
    target_var = tk.StringVar()
    target_dropdown = ttk.Combobox(root, textvariable=target_var, state="readonly", font=font_entry, width=30)
    target_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # Buttons for cleaning and classification
    tk.Button(root, text="Run Cleaning", command=run_cleaning, bg="#2196F3", fg="white", font=font_btn).grid(row=2, column=1, sticky="w", pady=10, padx=5)
    tk.Button(root, text="Run Classification", command=run_classification, bg="#4CAF50", fg="white", font=font_btn).grid(row=2, column=1, sticky="e", pady=10, padx=5)

    # Results text box
    results_box = scrolledtext.ScrolledText(root, width=110, height=20, wrap=tk.WORD, font=font_results)
    results_box.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    root.mainloop()
