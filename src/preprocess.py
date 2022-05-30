import argparse
import json
from pathlib import Path
import shutil

from rhoknp import KNP

from transcription import Transcription, parse_transcription_file

IMAGE_INTERVAL_MS = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', '-o', type=str, help='Path to output directory.')
    parser.add_argument('--image-dir', '--img', type=str, help='Path to the directory containing images.')
    parser.add_argument('--transcription-dir', '--text', type=str, help='Path to the file containing image ids.')
    parser.add_argument('--dialog-ids', '--id', type=str, help='Path to the file containing dialog ids.')
    # parser.add_argument('--lang', choices=['en', 'ja'], default='en', help='Language of the captions.')
    args = parser.parse_args()

    output_base_dir = Path(args.out_dir)
    image_base_dir = Path(args.image_dir)
    transcription_dir = Path(args.transcription_dir)
    knp = KNP()

    for dialog_id in Path(args.dialog_ids).read_text().splitlines():
        output_dir = output_base_dir / dialog_id
        output_dir.mkdir(exist_ok=True)
        output_image_dir = output_dir / 'images'
        output_image_dir.mkdir(exist_ok=True)
        image_dir = image_base_dir / dialog_id
        image_dir.mkdir(exist_ok=True)
        images = [(path, IMAGE_INTERVAL_MS * i) for i, path in enumerate(sorted(image_dir.glob('*.png')))]
        latest_file = sorted(transcription_dir.joinpath(dialog_id).glob(f'{dialog_id}-*.txt'), key=lambda p: p.name)[-1]
        transcriptions: list[Transcription] = parse_transcription_file(latest_file)
        infos: list[dict] = []
        knp_string = ''
        sidx = 0
        for transcription in transcriptions:
            raw_text = ''.join(str(token) for token in transcription.utterance)
            document = knp.apply_to_document(raw_text)
            for sentence in document.sentences:
                sentence.sid = f'{dialog_id}-{sidx:02}'
                sidx += 1
            knp_string += document.to_knp()
            images_within_utterance = [
                img for img in images if transcription.start_time <= img[1] < transcription.end_time
            ]
            for img in images_within_utterance:
                shutil.copy(img[0], output_image_dir / img[0].name)
            info = {
                'text': raw_text,
                'sids': [sentence.sid for sentence in document.sentences],
                'start': transcription.start_time,
                'end': transcription.end_time,
                'duration': transcription.duration,
                'speaker': transcription.speaker,
                'images': [
                    {
                        'id': img[0].stem,
                        'path': str((output_image_dir / img[0].name).relative_to(output_dir)),
                        'time': img[1]
                    }
                    for img in images_within_utterance
                ],
            }
            infos.append(info)
        output_dir.joinpath(f'{dialog_id}.knp').write_text(knp_string)
        output_dir.joinpath('info.json').write_text(json.dumps(infos, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
