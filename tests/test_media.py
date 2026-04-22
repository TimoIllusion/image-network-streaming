from inference_streaming_benchmark.frontend import media


def test_media():

    frame = next(media.generate_frames())

    assert type(frame) is bytes, "Camera could not generate frames."
