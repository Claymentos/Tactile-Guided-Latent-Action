import torch
import hydra
import pygame

from PIL import Image
from hydra.utils import call
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


@hydra.main(config_path="conf", config_name="test")
def main(cfg):
    model = call(cfg.lightning_module)
    model.eval()

    input_image = Image.open(cfg.image_path)    
    input_tensor = pil_to_tensor(input_image.resize((64, 64))) / 255.0
    input_video = input_tensor.unsqueeze(0).unsqueeze(0).to(model.device)

    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("LAM CoinRun")

    display_image = pygame.image.fromstring(input_image.tobytes(), input_image.size, input_image.mode)
    display_image = pygame.transform.scale(display_image, (512, 512))
    screen.blit(display_image, (0, 0))
    pygame.display.flip()

    running = True

    while(running):
        event = pygame.event.wait()
        
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            action = event.unicode

            with torch.no_grad():
                action_token = model.model.vq.project_out(model.model.vq.codebook[int(action)]).unsqueeze(0).unsqueeze(0).to(model.device)
                output = model.model.decode(input_video, action_token)
                output_image = (output[0][-1]*255).to('cpu', torch.uint8)
                
            output_image = to_pil_image(output_image)       
            input_video = torch.cat((input_video, output[:, -1].unsqueeze(1)), dim=1)

            if input_video.size(1) > 16:
                input_video = input_video[:, 1:, :, :, :]

            display_image = pygame.image.fromstring(output_image.tobytes(), output_image.size, output_image.mode)
            display_image = pygame.transform.scale(display_image, (512, 512))
            screen.blit(display_image, (0, 0))
            pygame.display.flip()


if __name__ == "__main__":
    main()