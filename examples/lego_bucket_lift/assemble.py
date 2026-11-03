"""Validate PMGS PNG frames and create a standalone MP4, GIF and keyframe sheet."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from common import PROJECT, protocol


def validate_frames(directory, indices, size):
    from PIL import Image
    expected = {'frame_%04d.png' % i for i in indices}
    found = {p.name for p in directory.glob('frame_*.png')}
    if found != expected:
        raise ValueError('Frame sequence differs: missing=%s extra=%s' % (sorted(expected-found), sorted(found-expected)))
    for name in sorted(expected):
        with Image.open(directory / name) as image:
            if image.size != (size, size):
                raise ValueError('Unexpected frame dimensions: ' + name)
            image.verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'animations' / 'lego_bucket_lift')
    parser.add_argument('--preview', action='store_true', help='Build a contact sheet from four preview frames')
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    config = protocol()
    output = args.output_root.expanduser().resolve()
    metadata = json.loads((output / ('preview_metadata.json' if args.preview else 'metadata.json')).read_text())
    indices = [0, 15, 29, 59] if args.preview else list(range(config['frame_count']))
    directory = output / ('preview' if args.preview else 'frames')
    if (metadata['frame_count'] != len(indices) or metadata['camera_c2w'] != config['camera_c2w']
            or metadata['controller_z'] != config['controller_z']):
        raise ValueError('Rendered frames use a different animation protocol')
    size = config['resolution']
    validate_frames(directory, indices, size)
    keys = [0, 15, 29, 59]
    tile = min(size, 400)
    sheet = Image.new('RGB', (2 * tile, 2 * (tile + 28)), '#15171b')
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(keys):
        x, y = (slot % 2) * tile, (slot // 2) * (tile + 28)
        draw.text((x+8, y+8), 'PMGS | frame %03d | %.2fs' % (index, index / config['fps']), fill='white')
        with Image.open(directory / ('frame_%04d.png' % index)) as image:
            sheet.paste(image.convert('RGB').resize((tile, tile)), (x, y+28))
    contact = output / ('preview_keyframes.jpg' if args.preview else 'keyframes.jpg')
    sheet.save(contact, quality=95)
    if args.preview:
        print(contact)
        return
    for binary in ['ffmpeg', 'ffprobe']:
        if shutil.which(binary) is None:
            raise RuntimeError(binary + ' is required on PATH')
    target = output / 'pmgs_bucket_lift.mp4'
    logs = output / 'assembly_logs'
    logs.mkdir(exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-framerate', str(config['fps']),
               '-i', str(directory / 'frame_%04d.png'), '-frames:v', str(config['frame_count']),
               '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    with (logs / 'encode.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                      '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(target)], text=True))
    stream = probe['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    if (int(stream['nb_read_frames']) != config['frame_count'] or stream['width'] != size
            or stream['height'] != size or numerator / denominator != config['fps']):
        raise RuntimeError('Encoded MP4 failed frame count, size or fps validation')
    gifs = []
    for index in range(0, config['frame_count'], 2):
        with Image.open(directory / ('frame_%04d.png' % index)) as image:
            gifs.append(image.convert('RGB').resize((min(size, 480), min(size, 480))))
    gif_path = output / 'pmgs_bucket_lift.gif'
    gifs[0].save(gif_path, save_all=True, append_images=gifs[1:], duration=round(2000 / config['fps']), loop=0, optimize=False)
    report = dict(status='passed', method='PMGS', frame_count=config['frame_count'], fps=config['fps'],
                  resolution=size, mp4=target.name, gif=gif_path.name, keyframes=contact.name, probe=probe)
    (output / 'assembly_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PMGS_MEDIA_COMPLETE', target)


if __name__ == '__main__':
    main()
