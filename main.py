from ultralytics import YOLO
import pandas as pd
import os

# ===== CẤU HÌNH =====
DATA_YAML = "D:\\IT\\CT201e\\datatest\\dataset\\data.yaml"
MODEL_NAME = "yolo26m.pt"
EPOCH = 250
OUTPUT_FILE = "result_yolo26m.csv"

def main():
    print(f"\n=== TRAIN {MODEL_NAME} | EPOCH = {EPOCH} ===\n")

    # load pretrained model
    model = YOLO(MODEL_NAME)

    # train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCH,
        patience=50,
        imgsz=768,
        batch=8,
        workers=0,
        plots=True,
        name=f"{MODEL_NAME.replace('.pt','')}_epoch_{EPOCH}",
        exist_ok=True
    )

    # đọc file kết quả
    df = pd.read_csv(
        os.path.join(results.save_dir, "results.csv")
    )

    # lấy epoch tốt nhất
    best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]

    # lưu kết quả
    summary = {
        "epoch": int(best_row["epoch"]),
        "mAP50-95": round(best_row["metrics/mAP50-95(B)"], 4),
        "mAP50": round(best_row["metrics/mAP50(B)"], 4),
        "precision": round(best_row["metrics/precision(B)"], 4),
        "recall": round(best_row["metrics/recall(B)"], 4),
    }

    result_df = pd.DataFrame([summary])
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Đã xuất file: {OUTPUT_FILE}")
    print(result_df)

if __name__ == "__main__":
    main()