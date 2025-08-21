'''
多线程读取文件夹内视频文件,叠加视频帧得到背景图
'''
import cv2
import numpy as np
import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

cpu_num = os.cpu_count() or 4  # 如果获取失败，默认用4线程

def save_image_with_chinese_path(path, img):
    ext = os.path.splitext(path)[1]  # 取扩展名，例如 ".jpg"
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        with open(path, mode="wb") as f:
            encoded_img.tofile(f)
    else:
        raise IOError(f"图像编码失败: {path}")
    
# ----------------- 视频背景提取函数 -----------------
def extract_background_from_video(video_path, frame_sample_sec=1, max_frames=200,
                                  stop_event=None, start_time=None, end_time=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25
    frame_interval = max(1, int(fps * frame_sample_sec))

    # 将开始和结束时间转换为帧索引
    start_frame_idx = int(fps * start_time) if start_time is not None else 0
    end_frame_idx = int(fps * end_time) if end_time is not None else float('inf')

    # 直接跳到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    accum = None
    count = 0
    frame_idx = start_frame_idx
    last_valid_frame = start_frame_idx

    while frame_idx <= end_frame_idx:
        if stop_event is not None and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 按采样间隔累加
        if (frame_idx - start_frame_idx) % frame_interval == 0:
            if accum is None:
                accum = np.zeros(frame.shape, dtype=np.float32)
            accum += frame.astype(np.float32)
            count += 1
            last_valid_frame = frame_idx  # 更新最后处理到的帧
            if count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    if count == 0:
        return None, None
    
    # 实际结束秒数 = 最后一帧 / fps
    actual_end_time = last_valid_frame / fps
    return (accum / count).astype(np.uint8), actual_end_time


# ----------------- 多线程视频处理 -----------------
def process_videos_worker_thread(video_queue, output_folder, frame_sample_sec, max_frames,
                                 processed_counter, processed_lock, total_count, ui_update_cb, stop_event,
                                 start_time, end_time):
    while True:
        if stop_event.is_set():
            break
        try:
            filename = video_queue.get_nowait()
        except queue.Empty:
            break

        video_path = filename   # 现在 queue 里放的是完整路径
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 状态更新（当前进度以 processed_counter 为基础估算）
        with processed_lock:
            curr_done = processed_counter[0]
        ui_update_cb("status", f"正在处理: {filename}", int(curr_done / max(1, total_count) * 100))
        ui_update_cb("log", f"开始处理: {filename}\n", 0)

        background, actual_end_time  = extract_background_from_video(video_path,
                                                                    frame_sample_sec, max_frames, stop_event,
                                                                    start_time, end_time)
        if background is not None and not stop_event.is_set():
            # 新逻辑：加上时间范围
            time_suffix = ""
            if start_time is not None or actual_end_time is not None:
                start_str = str(int(start_time)) if start_time is not None else ""
                end_str = str(int(actual_end_time)) if actual_end_time is not None else ""
                time_suffix = f"_{start_str}-{end_str}"
            output_path = os.path.join(output_folder, f"{base_name}_background{time_suffix}.jpg")
                
            try:
                save_image_with_chinese_path(output_path, background)
                ui_update_cb("log", f"完成保存: {output_path}\n", 0)
            except Exception as e:
                ui_update_cb("log", f"保存失败: {output_path}，错误: {e}\n", 0)
        else:
            ui_update_cb("log", f"跳过: {filename}\n", 0)

        # 更新已处理计数（保护）
        with processed_lock:
            processed_counter[0] += 1
            curr = processed_counter[0]
        progress_pct = int(curr / max(1, total_count) * 100)
        ui_update_cb("progress", "", progress_pct)

        video_queue.task_done()


def start_processing_thread(root, video_queue, output_folder, frame_sample_sec, max_frames,
                            start_btn, cancel_btn, progress_var, progress_label,
                            progress_label_percent, log_text, stop_event, 
                            start_time, end_time,
                            num_threads=cpu_num):
    """
    注意：这个函数会启动 worker 线程并返回一个 monitor 线程对象（daemon），
    主线程可以把它保存起来以便在退出时 join（示例中我们把返回值存入 worker_thread_holder）。
    """

    def ui_update(action, text, progress_pct):
        def _update():
            if action == "status":
                progress_label.config(text=text)
            elif action == "log" and text:
                log_text.insert(tk.END, text)
                log_text.see(tk.END)
            elif action == "progress":
                progress_var.set(progress_pct)
            percent = progress_var.get()
            progress_label_percent.config(text=f"{percent}%")
        root.after(0, _update)

    def finish_cb(total_done, total_videos):
        def _finish():
            start_btn.config(state=tk.NORMAL)
            cancel_btn.config(state=tk.DISABLED)
            progress_label.config(text="完成" if not stop_event.is_set() else "已取消")
            progress_label_percent.config(text=f"{progress_var.get()}%")
            messagebox.showinfo("完成", f"已处理 {total_done} / {total_videos} 个视频，结果保存在：\n{output_folder}")
        root.after(0, _finish)

    # 禁用开始按钮，启用取消按钮
    start_btn.config(state=tk.DISABLED)
    cancel_btn.config(state=tk.NORMAL)
    progress_var.set(0)
    progress_label.config(text="开始处理...")
    progress_label_percent.config(text="0%")
    log_text.delete("1.0", tk.END)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_counter = [0]
    processed_lock = threading.Lock()
    total_videos = video_queue.qsize()

    # 启动 worker 线程
    threads = []
    for _ in range(min(num_threads, total_videos)):
        t = threading.Thread(target=process_videos_worker_thread,
                             args=(video_queue, output_folder, frame_sample_sec, max_frames,
                                   processed_counter, processed_lock, total_videos, ui_update, stop_event,
                                   start_time, end_time),
                             daemon=True)
        t.start()
        threads.append(t)

    # 后台线程等待全部任务完成并回调 finish
    def monitor_threads():
        # 等待队列处理完或 stop_event 被设置
        while True:
            # 如果取消，则快速退出等待
            if stop_event.is_set():
                break
            all_done = all(not t.is_alive() for t in threads)
            if all_done:
                break
            # 小睡一下，避免 busy loop
            threading.Event().wait(0.2)

        # 确保所有线程 join（给出一定 timeout，避免卡死）
        for t in threads:
            t.join(timeout=1.0)
        # 最终回调
        with processed_lock:
            total_done = processed_counter[0]
        finish_cb(total_done, total_videos)

    monitor = threading.Thread(target=monitor_threads, daemon=True)
    monitor.start()
    return monitor



def build_gui():
    root = tk.Tk()
    root.title("视频背景提取工具")
    root.geometry("720x520")
    root.minsize(640, 420)

    # ===== Input row =====
    input_frame = tk.Frame(root)
    input_frame.pack(fill="x", padx=12, pady=(10, 6))

    # 文件输入行
    tk.Label(input_frame, text="输入视频文件:", width=12, anchor="w").grid(row=0, column=0, sticky="w")
    file_var = tk.StringVar()
    file_entry = tk.Entry(input_frame, textvariable=file_var)
    file_entry.grid(row=0, column=1, sticky="ew", padx=6)
    tk.Button(input_frame, text="浏览文件", width=10,
            command=lambda:browse_file()).grid(row=0, column=2, padx=6)

    # 文件夹输入行
    tk.Label(input_frame, text="输入视频文件夹:", width=12, anchor="w").grid(row=1, column=0, sticky="w")
    folder_var = tk.StringVar()
    folder_entry = tk.Entry(input_frame, textvariable=folder_var)
    folder_entry.grid(row=1, column=1, sticky="ew", padx=6)
    tk.Button(input_frame, text="浏览文件夹", width=10,
            command=lambda:browse_folder()).grid(row=1, column=2, padx=6)

    input_frame.grid_columnconfigure(1, weight=1)
    
    
    # ===== 选择文件或文件夹函数 =====
    def browse_file():
        path = filedialog.askopenfilename(title="选择视频文件",filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.h264")])
        if path:
            file_var.set(path)          # 设置文件输入框
            folder_var.set("")          # 清空文件夹输入框
            file_entry.config(state="normal")    # 保证自己可编辑
            folder_entry.config(state="disabled")  # 禁用文件夹输入框

    def browse_folder():
        path = filedialog.askdirectory(title="选择视频文件夹")
        if path:
            folder_var.set(path)        # 设置文件夹输入框
            file_var.set("")            # 清空文件输入框
            folder_entry.config(state="normal")  # 保证自己可编辑
            file_entry.config(state="disabled")   # 禁用文件输入框

    # ===== 定义互斥逻辑 =====
    def on_file_change(*args):
        val = file_var.get().strip()
        if val and file_entry.focus_get() == file_entry:
            folder_entry.config(state="disabled")   # 文件有值，文件夹禁用
        elif not val:
            folder_entry.config(state="normal")     # 文件清空，文件夹可用

    def on_folder_change(*args):
        val = folder_var.get().strip()
        if val and folder_entry.focus_get() == folder_entry:
            file_entry.config(state="disabled")     # 文件夹有值，文件禁用
        elif not val:
            file_entry.config(state="normal")       # 文件夹清空，文件可用

    file_var.trace_add("write", on_file_change)
    folder_var.trace_add("write", on_folder_change)

    # ===== Output row =====
    output_frame = tk.Frame(root)
    output_frame.pack(fill="x", padx=12, pady=(0, 6))

    tk.Label(output_frame, text="输出背景图文件夹:", width=16, anchor="w").grid(row=0, column=0, sticky="w")
    output_var = tk.StringVar()
    output_entry = tk.Entry(output_frame, textvariable=output_var)
    output_entry.grid(row=0, column=1, sticky="ew", padx=6)
    output_frame.grid_columnconfigure(1, weight=1)
    tk.Button(output_frame, text="浏览", width=10,
              command=lambda: output_var.set(filedialog.askdirectory())).grid(row=0, column=2, padx=6)

    # ===== Parameters =====
    param_frame = tk.LabelFrame(root, text="参数设置")
    param_frame.pack(fill="x", padx=12, pady=8)

    tk.Label(param_frame, text="采样间隔（秒）:").grid(row=0, column=0, padx=6, pady=8, sticky="w")
    interval_var = tk.StringVar(value="1")
    tk.Entry(param_frame, textvariable=interval_var, width=8).grid(row=0, column=1, sticky="w")

    tk.Label(param_frame, text="最大帧数:").grid(row=0, column=2, padx=6, pady=8, sticky="w")
    maxframes_var = tk.StringVar(value="200")
    tk.Entry(param_frame, textvariable=maxframes_var, width=8).grid(row=0, column=3, sticky="w")
    
    tk.Label(param_frame, text="开始时间（秒）:").grid(row=1, column=0, padx=6, pady=8, sticky="w")
    starttime_var = tk.StringVar(value="0")
    tk.Entry(param_frame, textvariable=starttime_var, width=8).grid(row=1, column=1, sticky="w")

    tk.Label(param_frame, text="结束时间（秒）:").grid(row=1, column=2, padx=6, pady=8, sticky="w")
    endtime_var = tk.StringVar(value="")  # 留空表示到视频结束
    tk.Entry(param_frame, textvariable=endtime_var, width=8).grid(row=1, column=3, sticky="w")

    # ===== Progress area =====
    status_frame = tk.Frame(root)
    status_frame.pack(fill="x", padx=12, pady=(6, 6))

    progress_label = tk.Label(status_frame, text="准备就绪", anchor="w")
    progress_label.grid(row=0, column=0, sticky="w")

    progress_label_percent = tk.Label(status_frame, text="0%")
    progress_label_percent.grid(row=0, column=1, sticky="e")

    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400,
                                   mode="determinate", variable=progress_var)
    progress_bar.pack(fill="x", padx=12, pady=(0, 8))

    # ===== Log window =====
    tk.Label(root, text="处理日志:").pack(anchor="w", padx=12)
    log_text = tk.Text(root, height=12)
    log_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    # ===== Buttons =====
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=12, pady=(0, 12))

    stop_event = threading.Event()
    worker_thread_holder = {"thread": None}  # mutable holder so we can refer later

    def on_start():
        input_file = file_var.get().strip()
        input_folder = folder_var.get().strip()
        output_folder = output_var.get().strip()
        
        # 确定使用哪一个输入
        input_path = input_file if input_file else input_folder
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请输入有效的视频文件或文件夹！")
            return
    
        try:
            frame_sample_sec = float(interval_var.get())
            max_frames = int(maxframes_var.get())
            start_time = float(starttime_var.get()) if starttime_var.get().strip() else None
            end_time = float(endtime_var.get()) if endtime_var.get().strip() else None
        except ValueError:
            messagebox.showerror("输入错误", "采样间隔、最大帧数和时间必须是数字！")
            return

        
        if not output_folder:
            output_folder = os.path.join(os.path.dirname(input_path), "backgrounds")
            output_var.set(output_folder)
            
        # 根据文件或文件夹获取视频文件列表
        video_files = gather_video_files(input_path)
        if not video_files:
            messagebox.showwarning("提示", "未找到可处理的视频文件")
            return    
        
        # 创建队列并启动线程
        video_queue = queue.Queue()
        for vf in video_files:
            video_queue.put(vf)
        
        # clear stop_event and start thread
        stop_event.clear()
        worker = start_processing_thread(root, video_queue, output_folder,
                                         frame_sample_sec, max_frames,
                                         start_btn, cancel_btn, progress_var,
                                         progress_label, progress_label_percent, log_text, stop_event,
                                         start_time, end_time)
        worker_thread_holder["thread"] = worker

    def on_cancel():
        if messagebox.askyesno("取消", "确定要取消当前处理吗？"):
            stop_event.set()
            cancel_btn.config(state=tk.DISABLED)
            progress_label.config(text="正在取消...")

    start_btn = tk.Button(btn_frame, text="开始处理", width=12, command=on_start)
    start_btn.pack(side="left", padx=6)

    cancel_btn = tk.Button(btn_frame, text="取消", width=12, command=on_cancel, state=tk.DISABLED)
    cancel_btn.pack(side="left", padx=6)

    def on_close():
        thr = worker_thread_holder.get("thread")
        if thr is not None and thr.is_alive():
            if not messagebox.askyesno("退出", "当前有任务在运行，确认退出并取消任务？"):
                return
            stop_event.set()
            thr.join(timeout=1.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    return root

# ===== 新增辅助函数 =====
def gather_video_files(path):
    """
    根据输入路径生成视频文件列表：
    - 如果 path 是文件，直接返回包含该文件的列表
    - 如果 path 是文件夹，递归收集所有视频文件
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".h264"):
            return [path]
        else:
            return []
    elif os.path.isdir(path):
        video_files = []
        for root_dir, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".h264")):
                    video_files.append(os.path.join(root_dir, f))
        return video_files
    return []


def main():
    root = build_gui()
    root.mainloop()


if __name__ == "__main__":
    main()
