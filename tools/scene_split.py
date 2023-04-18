import json
import nuscenes

from nuscenes.utils import splits

def main() -> None:
    day_scenes = None
    night_scenes = None
    json_path = "/homes/wlq20/CM30082/TransFusion-Environment/TransFusion/data/nuscenes/v1.0-mini/scene.json"
    with open(json_path) as json_file:
        scenes = json.load(json_file)
        
    if "v1.0-trainval" in json_path:
        day_scenes = [scene["name"] for scene in scenes if scene["name"] in splits.val and not "NIGHT" in scene["description"].upper()]
        night_scenes = [scene["name"] for scene in scenes if scene["name"] in splits.val and "NIGHT" in scene["description"].upper()]
    if "v1.0-mini" in json_path:
        day_scenes = [scene["name"] for scene in scenes if not "NIGHT" in scene["description"].upper()]
        night_scenes = [scene["name"] for scene in scenes if "NIGHT" in scene["description"].upper()]

    print(f"\nDay Scenes\n----------\n{day_scenes}\n")
    print(f"\nNight Scenes\n------------\n{night_scenes}\n")

if __name__ == "__main__":
    main()