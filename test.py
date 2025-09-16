import torch
import hydra
import pygame

from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate, call
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


class GameHandler:
    def __init__(self, max_action):
        self.input_buffer = ""
        self.max_action = max_action
        self.screen = pygame.display.set_mode((512, 512))
        pygame.display.set_caption("LAM CoinRun")

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.unicode.isdigit():
                self.input_buffer += event.unicode

            elif event.key == pygame.K_BACKSPACE:
                self.input_buffer = self.input_buffer[:-1]

            elif event.key == pygame.K_RETURN and self.input_buffer:
                number = int(self.input_buffer)
                self.input_buffer = ""

                if 0 <= number <= self.max_action:
                    return number
                else:
                    return None
                
        return None

    def display_image(self, image):
        display_image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        display_image = pygame.transform.scale(display_image, (512, 512))
        self.screen.blit(display_image, (0, 0))
        pygame.display.flip()


@hydra.main(config_path="conf", config_name="test")
def main(cfg):
    if (model_config_path := cfg.get("model_config")) is None:
        model_config_path = "/".join(cfg.lightning_module.checkpoint_path.split("/")[0:-1]) + f"/model_config.yaml{cfg.lightning_module.checkpoint_path.split('.ckpt')[-1]}"

    model_config = OmegaConf.load(model_config_path)
    backbone = instantiate(model_config.algo.model)
    model = call(cfg.lightning_module, model=backbone)
    model.freeze()

    input_image = Image.open(cfg.image_path).convert("RGB")
    input_tensor = pil_to_tensor(input_image.resize((model_config.algo.model.feature_extractor.image_size, model_config.algo.model.feature_extractor.image_size))) / 255.0
    input_video = input_tensor.unsqueeze(0).unsqueeze(0).to(model.device)

    handler = GameHandler(model.model.vq.codebook_size - 1)
    handler.display_image(input_image)

    running = True
    input_actions = None
    i = 0

    while(running):
        event = pygame.event.wait()
        
        if event.type == pygame.QUIT:
            running = False

        else:
            action = handler.handle_event(event)

            if action is not None:
                with torch.no_grad():
                    if hasattr(model.model.vq, "dequantize"): 
                        action_token = model.model.vq.dequantize([action]).unsqueeze(0).to(model.device)
                    else:
                        action_token = model.model.vq.project_out(model.model.vq.codebook[[action]]).unsqueeze(0)                  # Backward compatability
                    
                    if input_actions is None:
                        input_actions = action_token
                    else:
                        input_actions = torch.cat((input_actions, action_token), dim=1)

                    input_features = model.model.feature_extractor(input_video)
                    output = model.model.decode(input_features, input_actions)
                    output = model.model.feature_extractor._post_process(output)
                    output_image = (output[0][-1]*255).to('cpu', torch.uint8)
                    
                output_image = to_pil_image(output_image)   
                
                if cfg.save_images:
                    output_image.save(f"model_output_{i}.png")
                    i += 1

                input_video = torch.cat((input_video, output[:, -1].unsqueeze(1)), dim=1)

                if input_video.size(1) >= model_config.data.data_module.window_size:
                    input_video = input_video[:, 1:, :, :, :]
                    input_actions = input_actions[:, 1:, :]

                handler.display_image(output_image)



if __name__ == "__main__":
    main()