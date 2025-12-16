import cv2
import torch
import threading
import queue
import time

# 1. Setup Queues for thread-safe communication
input_queue = queue.Queue(maxsize=1) # Limit size to 1 to process only the latest frame
output_queue = queue.Queue(maxsize=1)

# A flag to signal the worker thread to stop
stop_event = threading.Event()

# 2. Define the PyTorch Worker Thread
class PyTorchWorker(threading.Thread):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.daemon = True # Allows the program to exit even if this thread is running

    def run(self):
        print("Worker thread started")
        # Set the number of threads for PyTorch to avoid CPU oversubscription
        # (important if OpenCV also uses multiple threads internally)
        torch.set_num_threads(1)

        while not stop_event.is_set():
            try:
                # Get the latest frame (and remove older ones if they accumulate)
                frame = input_queue.get(timeout=0.1)

                # Perform inference (this is CPU intensive)
                # Ensure model operations are thread-safe (inference generally is)
                results = self.model(frame)

                # Place results in the output queue
                if not output_queue.empty():
                    try:
                        output_queue.get_nowait() # Clear old result
                    except queue.Empty:
                        pass
                output_queue.put(results)

            except queue.Empty:
                continue # No frame available, keep checking
            except Exception as e:
                print(f"Error in worker thread: {e}")

# 3. Main OpenCV Event Loop
def main_opencv_loop(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Start the background worker thread
    worker = PyTorchWorker(model)
    worker.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Put the current frame in the input queue (overwrites previous if full)
        # CHANGED: move copy after the queue
        #     There's no way the garbage collector makes enough gains to justify
        #         multiple copies per poll
        if not input_queue.full():
            input_queue.put(frame.copy())

        # Pull results from the output queue if available
        if not output_queue.empty():
            results = output_queue.get_nowait()
            # You can now process and display 'results' (e.g., draw boxes)
            # This is where you integrate the ML output into the displayed frame
            # Example: display text indicating inference happened
            cv2.putText(frame, "Model Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('CV2 Feed with PyTorch Inference', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() # Signal worker to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    worker.join() # Wait for the worker thread to finish safely

if __name__ == '__main__':
    # Load your PyTorch model here (on CPU or GPU)
    # model = YourModelClass()
    # model.load_state_dict(torch.load('your_model.pth'))
    # model.eval()

    # Use a placeholder or a simple actual model for testing
    class PlaceholderModel:
        def __call__(self, frame):
            # Simulate processing time
            time.sleep(0.1)
            return "Simulated Results"

    model = PlaceholderModel()

    main_opencv_loop(model)
