import torch
import os
import sys
sys.path.insert(0, '.')

if __name__ == '__main__':
    from src.dataset import get_loaders
    from src.unet import UNet
    from src.plain_encoder import PlainEncoderDecoder
    from src.train import train_model
    from src.evaluate import plot_comparison, plot_predictions

    os.makedirs('models', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    train_loader, val_loader = get_loaders(batch_size=8)

    print("\n=== Training U-Net ===")
    unet = UNet(in_channels=3, num_classes=7)
    unet_history = train_model(unet, train_loader, val_loader,
                               epochs=30, lr=1e-3, device=device)
    torch.save(unet.state_dict(), 'models/unet.pth')

    print("\n=== Training Plain Encoder-Decoder baseline ===")
    plain = PlainEncoderDecoder(in_channels=3, num_classes=7)
    plain_history = train_model(plain, train_loader, val_loader,
                                epochs=30, lr=1e-3, device=device)
    torch.save(plain.state_dict(), 'models/plain_encoder.pth')

    plot_comparison(unet_history, plain_history)
    plot_predictions(unet, val_loader, device)

    unet.eval()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        unet, dummy, 'models/unet.onnx',
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print("Exported models/unet.onnx")