import urllib.request, tarfile, pathlib, sys, logging

local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from wn12 import balls
try:
    from ds12 import read_balls
except FileNotFoundError as e:
    logging.warning(e)
finally:
    sys.path.pop(0)

# by using this link you are subject to the ImageNet terms and conditions
# please formally request access via image-net.org before redistribution
# open source can be sustainable through integrity
base_url = "https://image-net.org/data/winter21_whole"
filename = lambda synset: f"{synset}.tar"
synset_url = lambda synset: f"{base_url}/{synset}.tar"
sources = [
        { "url": synset_url(synset), "tar": filename(synset), "dir": readable }
        for _, synset, readable in balls]

data_dir = local / "assets" / "data"
download_dir = data_dir / "downloads"

data_dir.mkdir(parents=True, exist_ok=True)
download_dir.mkdir(parents=True, exist_ok=True)

for info in sources:
    dest = data_dir / info["dir"]
    if dest.is_dir():
        continue
    tar = download_dir / info["tar"]
    if not tar.is_file():
        print("downloading", info["url"])
        urllib.request.urlretrieve(info["url"], tar)
    with tarfile.open(tar) as fp:
        fp.extractall(dest, filter="data")

# import nltk; nltk.download()
ids21 = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt"
def wn_balls():
    ids21path = download_dir / ids21.rsplit("/", 1)[-1]
    if not ids21path.is_file():
        print("downloading", ids21)
        urllib.request.urlretrieve(ids21, ids21path)

    from nltk.corpus import wordnet as wn

    wn21 = lambda nid: wn.synset_from_pos_and_offset(nid[0], int(nid[1:]))
    wn12 = [wn21(synset) for _, synset, _ in balls]
    hyper = wn21('n02778669')
    balls21 = []

    with open(ids21path) as fp:
        for synset in fp:
            if hyper in wn21(synset).hypernyms():
                balls21.append(synset.strip())
    return balls21, [wn21(i) for i in balls21]

# balls21 = wn_balls()
balls21 = [
        'n02799071', 'n02802426', 'n02839351', 'n02861147', 'n02881546',
        'n02882301', 'n03131967', 'n03134739', 'n03333252', 'n03378765',
        'n03445777', 'n03482877', 'n03589672', 'n03632100', 'n03721047',
        'n03742019', 'n03825442', 'n03942813', 'n03978575', 'n03982232',
        'n04023962', 'n04039742', 'n04113316', 'n04118538', 'n04254680',
        'n04256891', 'n04292221', 'n04409515', 'n04540053', 'n04584056']
