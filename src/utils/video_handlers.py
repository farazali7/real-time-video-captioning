import numpy as np
import cv2, argparse

from src.utils.frame_sampling_methods import play_video_from_frames


def get_video_frames(video_path: str) -> np.ndarray:
    """
    Get Video Frames:
    - Read video from the given path.
    - Return the frames of the video as a numpy array.
    - Simply reads the video and returns the frames as a numpy array.
    - Quick if there is no need to alter the frames.
    Args:
    - video_path: str: Path to the video file.
    Returns:
    - frames: np.ndarray: Frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> frames = get_video_frames(video_path)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    return np.array(frames)


def get_evenly_sampled_frames(video_path: str, num_frames: int) -> np.ndarray:
    """
    Get Video Frames evenly sampled across entire video:
    - Read video from the given path.
    - Return the frames of the video as a numpy array.
    - Simply reads the video and returns the frames as a numpy array.
    - Quick if there is no need to alter the frames.
    Args:
    - video_path: str: Path to the video file.
    - num_frames: Total number of frames to sample
    Returns:
    - frames: np.ndarray: Frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> frames = get_video_frames(video_path)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    indices = np.arange(0, frame_count, frame_count // num_frames, dtype=np.int32)[:num_frames]
    frames = []

    for i in range(indices[-1]+1):
        if i in indices:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        else:
            ret = video.grab()
            if not ret:
                break

    return np.array(frames)

def get_evenly_sampled_frames2(video_path: str, num_frames: int) -> np.ndarray:
    """
    Get Video Frames evenly sampled across entire video:
    - Read video from the given path.
    - Return the frames of the video as a numpy array.
    - Simply reads the video and returns the frames as a numpy array.
    - Quick if there is no need to alter the frames.
    Args:
    - video_path: str: Path to the video file.
    - num_frames: Total number of frames to sample
    Returns:
    - frames: np.ndarray: Frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> frames = get_video_frames(video_path)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    indices = np.arange(0, frame_count, frame_count // num_frames, dtype=np.int32)[:num_frames]
    frames = []

    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        res, frame = video.read()
        frames.append(frame)

    return np.array(frames)


def get_video_frames_with_resize(video_path: str, width_resize_ratio: float, height_resize_ratio: float) -> np.ndarray:
    """
    Get Video Frames with Resize:
    - Read video from the given path.
    - Resize the frames of the video.
    - Return the resized frames of the video as a numpy array.
    - Resizes the frames of the video and returns them as a numpy array along the width and height.
    - The integer values are taken for the resized width and height to resize to avoid errors.
    Args:
    - video_path: str: Path to the video file.
    - width_resize_ratio: float: Ratio to resize the width of the frames.
    - height_resize_ratio: float: Ratio to resize the height of the frames.
    Returns:
    - frames: np.ndarray: Resized frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> width_resize_ratio = 0.5
    >>> height_resize_ratio = 0.5
    >>> frames = get_video_frames_with_resize(video_path, width_resize_ratio, height_resize_ratio)
    >>> frames.shape
    (num_frames, downsampled_height, downsampled_width, 3)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=width_resize_ratio, fy=height_resize_ratio)
        frames.append(frame)

    return np.array(frames)


def get_video_frames_RGB_to_GRAY(video_path: str) -> np.ndarray:
    """
    Get Video Frames RGB to GRAY:
    - Read video from the given path.
    - Convert the frames of the video from RGB to GRAY.
    - Return the frames of the video as a numpy array.
    - Converts the frames of the video from RGB to GRAY and returns them as a numpy array.
    Args:
    - video_path: str: Path to the video file.
    Returns:
    - frames: np.ndarray: Frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> frames = get_video_frames_RGB_to_GRAY(video_path)
    >>> frames.shape
    (num_frames, height, width)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)

    return np.array(frames)


def get_video_frames_with_resize_RGB_to_GRAY(video_path: str, width_resize_ratio: float, height_resize_ratio: float) -> np.ndarray:
    """
    Get Video Frames with Resize RGB to GRAY:
    - Read video from the given path.
    - Resize the frames of the video.
    - Convert the frames of the video from RGB to GRAY.
    - Return the resized frames of the video as a numpy array.
    - Resizes the frames of the video and converts them from RGB to GRAY and returns them as a numpy array along the width and height.
    - The integer values are taken for the resized width and height to resize to avoid errors.
    Args:
    - video_path: str: Path to the video file.
    - width_resize_ratio: float: Ratio to resize the width of the frames.
    - height_resize_ratio: float: Ratio to resize the height of the frames.
    Returns:
    - frames: np.ndarray: Resized frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> width_resize_ratio = 0.5
    >>> height_resize_ratio = 0.5
    >>> frames = get_video_frames_with_resize_RGB_to_GRAY(video_path, width_resize_ratio, height_resize_ratio)
    >>> frames.shape
    (num_frames, downsampled_height, downsampled_width)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=width_resize_ratio, fy=height_resize_ratio)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)

    return np.array(frames)


def get_video_frames_with_downsampling(video_path: str, downsampling_factor: int) -> np.ndarray:
    """
    Get Video Frames with Downsampling:
    - Read video from the given path.
    - Downsample the frames of the video.
    - Return the downsampled frames of the video as a numpy array.
    - Downsamples the frames of the video and returns them as a numpy array.
    - The downsampling factor is used to downsample the frames.
    - This is different as it maintains the aspect ratio of the frames which retains meaning.
    - Equivalent to the get_video_frames_with_resize function but with a downsampling factor being the same for width and height.
    Args:
    - video_path: str: Path to the video file.
    - downsampling_factor: int: Downsampling factor for the frames.
    Returns:
    - frames: np.ndarray: Downsampled frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> downsampling_factor = 2
    >>> frames = get_video_frames_with_downsampling(video_path, downsampling_factor)
    >>> frames.shape
    (num_frames, downsampled_height, downsampled_width, 3)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=1/downsampling_factor, fy=1/downsampling_factor)
        frames.append(frame)

    return np.array(frames)


def get_video_frames_with_feature_enhancements(video_path: str, method: str = "histogram_equalization") -> np.ndarray:
    """
    Get Video Frames with Feature Enhancements:
    - Read video from the given path.
    - The goal is to have each frame with a higher resolution and more clarity.
    - The frames do not have their resolution changed, but the clarity is increased.
    - Return the frames of the video as a numpy array.
    - The method parameter is used to select the method to enhance the resolution.
    - The method parameter currently supports: 
    -- Method: "image_filtering": Gaussian Blur -> Laplacian Filter -> Sharpened and Deblurred Image
    -- Method: "histogram_equalization": Histogram Equalization
    -- Method: "unsharp_masking": Unsharp Masking
    -- Method: "contrast_stretching": Contrast Stretching
    Args:
    - video_path: str: Path to the video file.
    - method: str: Method to enhance the resolution of the frames (default: "histogram_equalization").
    Returns:
    - frames: np.ndarray: Frames of the video.

    Example:
    >>> video_path = "path/to/video.mp4"
    >>> method = "image_filtering"
    >>> frames = get_video_frames_with_feature_enhancements(video_path, method)
    >>> frames.shape
    (num_frames, height, width, 3)
    """
    video = cv2.VideoCapture(video_path)

    frames = []

    if method == "image_filtering":
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.Laplacian(frame, cv2.CV_64F)
            frame = cv2.convertScaleAbs(frame)
            frames.append(frame)
    elif method == "histogram_equalization":
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.equalizeHist(frame)
            frames.append(frame)
    elif method == "unsharp_masking":
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.addWeighted(frame, 1.5, frame, -0.5, 0)
            frames.append(frame)
    elif method == "contrast_stretching":
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            frames.append(frame)
    else:
        raise ValueError("The method parameter is not supported. Please select a supported method.")

    return np.array(frames)


def main(video_path: str):
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
    >>> python utils\video_handlers.py --function get_video_frames
    // This will run the get_video_frames function and display the sampled frames as a video.
    >>> python utils\video_handlers.py --function get_video_frames_with_resize
    // This will run the get_video_frames_with_resize function and display the sampled frames as a video.
    >>> python utils\video_handlers.py --function get_video_frames_RGB_to_GRAY
    // This will run the get_video_frames_RGB_to_GRAY function and display the sampled frames as a video.
    """
    try:
        video = cv2.VideoCapture(video_path)
        assert video.isOpened()
        video.release()
    except:
        raise Exception("Video Path does not exist.")

    parser = argparse.ArgumentParser(description="Frame Sampling Methods")
    parser.add_argument("--function", type=str, help="Function Name", default=".")
    args = parser.parse_args()

    if args.function == "get_video_frames":
        frames = get_video_frames(video_path)
        play_video_from_frames(frames, fps = 30)
    elif args.function == "get_video_frames_with_resize":
        width_resize_ratio = 0.5
        height_resize_ratio = 0.75
        frames = get_video_frames_with_resize(video_path, width_resize_ratio, height_resize_ratio)
        play_video_from_frames(frames, fps = 30)
    elif args.function == "get_video_frames_RGB_to_GRAY":
        frames = get_video_frames_RGB_to_GRAY(video_path)
        play_video_from_frames(frames, fps = 30)
    elif args.function == "get_video_frames_with_resize_RGB_to_GRAY":
        width_resize_ratio = 0.5
        height_resize_ratio = 0.75
        frames = get_video_frames_with_resize_RGB_to_GRAY(video_path, width_resize_ratio, height_resize_ratio)
        play_video_from_frames(frames, fps = 30)
    elif args.function == "get_video_frames_with_downsampling":
        downsampling_factor = 2
        frames = get_video_frames_with_downsampling(video_path, downsampling_factor)
        play_video_from_frames(frames, fps = 30)
    elif args.function == "get_video_frames_with_feature_enhancements":
        method = "contrast_stretching"
        frames = get_video_frames_with_feature_enhancements(video_path, method)
        play_video_from_frames(frames, fps = 30)
    else:
        frames = get_video_frames(video_path)
        play_video_from_frames(frames, fps = 30)


if __name__ == "__main__":
    """
    Main Function:
    - Test the video handlers functions.
    - Be sure to have a video file in the path (data/videos/file.mp4).
    - Have the current working directory as real-time-video-captioning (root).

    Issues:
    - The play_video_from_frames function is elusive because it converts the frames back to RGB.
    - The frames are converted to RGB before playing the video.
    - Can only really test the resizing and downsampling functions.

    >>> pwd
    .../real-time-video-captioning
    """
    video_path = "data/videos/test_2.mp4"
    main(video_path)