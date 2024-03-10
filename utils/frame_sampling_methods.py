import cv2, typing, warnings, argparse
import numpy as np
from sklearn.cluster import KMeans


D_TYPE = np.uint8
RANDOM_STATE = 42


def play_video_from_frames(frames: np.ndarray, fps: int) -> None:
    """
    Play Video from Frames:
    - Play the video from the given frames.
    - The frames will be played at the given frames per second (fps).
    - The video will be displayed in a window.
    - Makes sense to do this only when the subsampled frames are still in good number.
    Args:
    - frames: np.ndarray, shape (num_frames, height, width, 3), frames to play as a video.
    - fps: int, frames per second.
    Returns:
    - None.
    - Displays the video from the given frames.

    Example:
    >>> frames = np.random.randint(0, 255, size=(100, 240, 320, 3), dtype=np.uint8)
    >>> frames.shape
    (100, 240, 320, 3)
    >>> fps = 30
    >>> play_video_from_frames(frames, fps)
    """
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def uniform_sampling(video_path: str, retention_rate: float) -> np.ndarray:
    """
    Uniform Sampling:
    - Uniformly sample frames from the video.
    - Given a retention rate, uniformly sample frames from the video throughout the entire video.
    - Retention rate is the percentage of frames to retain from the original video.
    - For example, a retention rate of 0.5 means that 50% of the frames will be retained.
    - The retained frames will be used for captioning.
    Args:
    - video_path: str, path to the video.
    - retention_rate: float, percentage of frames to retain from the original video.
    Returns:
    - frames: np.ndarray, shape (num_frames, height, width, 3), uniformly sampled frames from the video.

    Example:
    >>> video_path = 'path/to/video.mp4'
    >>> retention_rate = 0.5
    >>> frames = uniform_sampling(video_path, retention_rate)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    num_retained_frames = int(num_frames * retention_rate)
    sampling_interval = num_frames // num_retained_frames

    retained_frames = []

    for i in range(num_frames):
        ret, frame = video.read()
        if ret and i % sampling_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            retained_frames.append(frame)

    frames = np.array(retained_frames, dtype=D_TYPE)
    
    video.release()
    return frames


def random_sampling_from_bins(video_path: str, num_bins: int) -> np.ndarray:
    """
    Random Sampling from Bins:
    - The video is first divided into bins.
    - Each bin contains a certain number of frames. 
    - The number of frames in each bin is determined by the number of bins.
    - Randomly sample one frame from each bin.
    - The retained frames will be used for captioning.
    - The more bins there are, the more frames will be retained, which will increase the diversity of the frames.
    - Ensures that the frames that are sampled are a subsequence of the original video.
    Args:
    - video_path: str, path to the video.
    - num_bins: int, number of bins to divide the video into.
    Returns:
    - frames: np.ndarray, shape (num_frames, height, width, 3), randomly sampled frames from the bins that are generated from the video.
    
    Example:
    >>> video_path = 'path/to/video.mp4'
    >>> num_bins = 10
    >>> frames = random_sampling_from_bins(video_path, num_bins)
    >>> frames.shape
    (num_bins, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_bin = num_frames // num_bins

    retained_frames = []

    for i in range(num_bins):
        bin_start = i * frames_per_bin
        bin_end = (i + 1) * frames_per_bin

        bin_frames = []
        for j in range(bin_start, bin_end):
            video.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bin_frames.append(frame)

        bin_frames = np.array(bin_frames, dtype=D_TYPE)

        if len(bin_frames) > 0:
            random_frame_index = np.random.choice(len(bin_frames))
            random_frame = bin_frames[random_frame_index]
            retained_frames.append(random_frame)

    frames = np.array(retained_frames, dtype=D_TYPE)
    
    video.release()
    return frames


def clustered_sampling(video_path: str, num_classes: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Clustered Sampling:
    - First downsample the frames from the video to a smaller dimension for easy clustering.
    - Use unsupervised learning to cluster the frames into a certain number of classes.
    - Classify the frames from the video into a certain number of classes (say k).
    - All the frames from the input video are divided into k classes using unsupervised learning.
    - The array of frames of the original video is now tracked along with its assigned class and this is separately maintained.
    - The original video is sequentially traversed.
    - Now, iteratively if the next frame belongs to a different class than the previous frame, only then the next frame is sampled and current class gets updated.
    - If the next frame belongs to the same class as the previous frame, then the next frame is not sampled.
    - The number of frames that are output depends on how frequently is the content changing in the video.
    Args:
    - video_path: str, path to the video.
    - num_classes: int, number of classes to divide the frames into. More classes will result in more frames and information being retained.
    Returns:
    - frames: np.ndarray, shape (num_frames, height, width, 3), frames that are sampled on the logic above from the video.
    - classes: np.ndarray, shape (num_frames,), classes assigned to the frames.

    Example:
    >>> video_path = 'path/to/video.mp4'
    >>> num_classes = 10
    >>> frames, classes = clustered_sampling(video_path, num_classes)
    >>> frames.shape
    (num_frames, height, width, 3)
    >>> classes.shape
    (num_frames,)
    """
    video = cv2.VideoCapture(video_path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    transformed_frames, original_frames = [], []

    for i in range(num_frames):
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame)
            transformed_frame = cv2.resize(frame, (32, 32))
            transformed_frame = transformed_frame.flatten()
            transformed_frames.append(transformed_frame)

    transformed_frames = np.array(transformed_frames, dtype=D_TYPE)

    kmeans = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE)
    classes = kmeans.fit_predict(transformed_frames)

    retained_frames = [original_frames[0]]
    retained_classes = [classes[0]]

    for i in range(1, num_frames):
        if classes[i] != classes[i - 1]:
            retained_frames.append(original_frames[i])
            retained_classes.append(classes[i])

    frames = np.array(retained_frames, dtype=D_TYPE)
    classes = np.array(retained_classes, dtype=D_TYPE)

    video.release()
    return frames, classes

def frame_difference_sampling(video_path: str, threshold: int) -> np.ndarray:
    """
    Frame Difference Sampling:
    - The video is sequentially traversed.
    - The first frame is always sampled.
    - The next frame that will be sampled will only be samples if the difference between the current frame and the previous frame is greater than a certain threshold.
    - The reference frame is updated to the current frame if the difference is greater than the threshold.
    - The number of frames that are output depends on how frequently is the content changing in the video.
    Args:
    - video_path: str, path to the video.
    - threshold: int, threshold for the difference between the frames.
    Returns:
    - frames: np.ndarray, shape (num_frames, height, width, 3), frames that are sampled on the logic above from the video.

    Example:
    >>> video_path = 'path/to/video.mp4'
    >>> threshold = 100
    >>> frames = frame_difference_sampling(video_path, threshold)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    retained_frames = [video.read()[1]]

    for i in range(1, num_frames):
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            previous_frame = retained_frames[-1]
            difference = cv2.absdiff(previous_frame, frame)
            difference = np.sum(difference)
            if difference > threshold:
                retained_frames.append(frame)

    frames = np.array(retained_frames, dtype=D_TYPE)
    
    video.release()
    return frames


def main(video_path: str) -> None:
    """
    Main Function:
    - For testing the functions in this file.
    - Checks if the video path provided exists.
    - Checks for the system arguments and runs the function accordingly.
    - If the system arguments are provided, then the main function will run the function present along with its given name.
    - If the system arguments are not provided, then the main function will run all the functions present along with its default name.

    Example:
    (First you need to be at the root directory to run the cmd executable commands)
    >>> pwd
    .../real-time-video-captioning
    >>> python real-time-video-captioning\utils\frame_sampling_methods.py --function frame_difference_sampling
    // This will run the frame_difference_sampling function and display the sampled frames as a video.
    >>> python real-time-video-captioning\utils\frame_sampling_methods.py --function uniform_sampling
    // This will run the uniform_sampling function and display the sampled frames as a video.
    """
    try:
        video = cv2.VideoCapture(video_path)
        assert video.isOpened()
        video.release()
    except:
        raise Exception("Video Path does not exist.")

    parser = argparse.ArgumentParser(description="Frame Sampling Methods")
    parser.add_argument("--function", type=str, help="Function Name", default="all")
    args = parser.parse_args()

    if args.function == "uniform_sampling":
        retention_rate = 0.5
        frames = uniform_sampling(video_path, retention_rate)
        print(frames.shape)
        play_video_from_frames(frames, 30)

    elif args.function == "random_sampling_from_bins":
        num_bins = 30
        frames = random_sampling_from_bins(video_path, num_bins)
        print(frames.shape)
        play_video_from_frames(frames, 10)

    elif args.function == "clustered_sampling":
        num_classes = 25
        frames, _ = clustered_sampling(video_path, num_classes)
        print(frames.shape)
        play_video_from_frames(frames, 10)

    elif args.function == "frame_difference_sampling":
        threshold = 1000
        frames = frame_difference_sampling(video_path, threshold)
        print(frames.shape)
        play_video_from_frames(frames, 30)
    
    else:
        retention_rate = 0.5
        frames = uniform_sampling(video_path, retention_rate)
        print(frames.shape)
        play_video_from_frames(frames, 30)

        num_bins = 30
        frames = random_sampling_from_bins(video_path, num_bins)
        print(frames.shape)
        play_video_from_frames(frames, 10)

        num_classes = 25
        frames, _ = clustered_sampling(video_path, num_classes)
        print(frames.shape)
        play_video_from_frames(frames, 10)

        threshold = 1
        frames = frame_difference_sampling(video_path, threshold)
        print(frames.shape)
        play_video_from_frames(frames, 30)


if __name__ == "__main__":
    """
    For testing the functions in this file.
    - Provide the path to the video that you want to test the functions on (be sure to include this from the root).
    - Each function should be able to return the frames that are sampled from the video.
    - Depending on the number of frames being returned, you can choose to play the subframes as a video or sequence of images.
    """
    warnings.filterwarnings("ignore")

    # Selecting a video (keep the video in the data/videos folder)
    video_path = 'real-time-video-captioning/data/videos/test_1.mp4'

    # Testing the functions
    main(video_path = video_path)