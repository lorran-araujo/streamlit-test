import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Carregar os pesos do modelo
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 é o tamanho após os pools
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('mnist_cnn_pytorch.pth', map_location=torch.device('cpu')))
model.eval()  # Modo de avaliação

# 2. Interface Streamlit
st.title("Reconhecimento de Dígitos Manuscritos")
st.write("Envie uma imagem de um dígito (qualquer tamanho/cor):")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 3. Pré-processamento da imagem
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", width=200)

    # Converter para escala de cinza e redimensionar para 28x28
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Garante 1 canal (MNIST)
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mesma normalização do treino
    ])

    try:
        image_tensor = transform(image).unsqueeze(0)  # Adiciona dimensão do batch (1, 1, 28, 28)

        # 4. Inferência
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output).item()
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100

        # 5. Exibir resultados
        st.write(f"## Predição: **{prediction}**")
        st.write("### Probabilidades por classe:")
        for i, prob in enumerate(probabilities):
            st.write(f"Dígito {i}: {prob:.1f}%")

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")