<p align="center">
  <img src="docs/header.svg" width="100%" alt="IRIS — Estação de Leitura Óptica"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-FFB454?style=flat-square&logo=python&logoColor=0B0E13&labelColor=151B24" alt="Python"/>
  <img src="https://img.shields.io/badge/PyQt6-6.x-FFB454?style=flat-square&logo=qt&logoColor=0B0E13&labelColor=151B24" alt="PyQt6"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-FFB454?style=flat-square&logo=opencv&logoColor=0B0E13&labelColor=151B24" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-✓-4ADE80?style=flat-square&logoColor=0B0E13&labelColor=151B24" alt="ONNX"/>
  <img src="https://img.shields.io/badge/Windows-10%2F11-8FA8BF?style=flat-square&logo=windows&logoColor=EAF0F6&labelColor=151B24" alt="Windows"/>
</p>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

<table>
<tr>
<td width="52%" valign="top">

### Visão Geral

**Iris** é uma estação de leitura óptica de alta performance para decodificação de QR codes em tempo real via webcam. O sistema opera como um instrumento de precisão — a interface é uma metáfora direta de uma objetiva de câmera, onde o diafragma de íris comunica o estado do sistema através de animação física.

Construído inteiramente em **PyQt6** com renderização nativa, sem dependências web. Cada frame de câmera passa por um pipeline de detecção YOLO via ONNX Runtime antes de chegar ao decodificador.

</td>
<td width="48%" valign="top">

### Recursos

```
◈  Detecção por IA          YOLO · ONNX Runtime
◈  Feed ao vivo             60 FPS · OpenCV
◈  Linha de varredura       Animação âmbar contínua
◈  Campo de estrelas        75 partículas interativas
◈  Diafragma animado        7 lâminas · física real
◈  Histórico de leituras    Flash fósforo · timestamp
◈  Modo dev (IRIS_DEVMODE)  Sem writes em banco
◈  Ícone na barra de tarefas AppUserModelID nativo
```

</td>
</tr>
</table>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

## Paleta Visual

<p align="center">
  <img src="docs/palette.svg" width="100%" alt="Paleta de cores do Iris"/>
</p>

<br/>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

## Arquitetura

```
desk_Iris/
├── app/
│   ├── src/
│   │   ├── assets/
│   │   │   └── img/           ◈  logo.png · logo.ico · logo-com-fundo.png
│   │   ├── models/
│   │   │   └── route_verify/  ◈  best.onnx · calibration/
│   │   ├── services/          ◈  camada de serviços — adapters externos
│   │   └── UIX/
│   │       ├── components/
│   │       │   ├── shared.py      ◈  paleta C · IrisAperture · IrisButton · IrisAppBar
│   │       │   └── star_field.py  ◈  StarFieldPanel · 75 partículas · repulsão por mouse
│   │       ├── main_menu/
│   │       │   └── view.py        ◈  menu principal · cards animados
│   │       └── modules/
│   │           └── decoding/
│   │               └── live_qr/   ◈  feed · linha de varredura · histórico
│   └── main.py
└── docs/                          ◈  assets do README
```

<br/>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

## Animações do Sistema

<table>
<tr>
<td width="33%" valign="top">

**Diafragma de Íris**

Simulação física de 7 lâminas. O estado do sistema é comunicado através da abertura:

```
FECHADO    openness = 0.10   idle
RESPIRANDO openness 0.15↔0.70 init
ABERTO     openness = 0.78   live
```
Easing `InOutCubic` · 650ms

</td>
<td width="33%" valign="top">

**Campo de Estrelas**

75 partículas com deriva a 35°. Interagem com o cursor dentro de 130px — linhas de constelação em âmbar aparecem ao aproximar o mouse.

```python
STAR_COUNT    = 75
DRIFT_ANGLE   = 35°
REPEL_RADIUS  = 130 px
FPS           = 60
VELOCITY_DECAY= 0.91
```

</td>
<td width="33%" valign="top">

**Linha de Varredura**

Gradiente âmbar que percorre o feed ao vivo sinalizando atividade de captura.

```
#FFB454  →  gradiente  →  #FFB454
opacity   40 px halo
step      0.004 / 16ms
reticle   4 cantos · âmbar
```

</td>
</tr>
</table>

<br/>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

## Instalação

**Pré-requisitos:** Python 3.11+ · webcam · Windows 10/11

```bash
# 1. Clone o repositório
git clone <repo-url>
cd desk_Iris

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute
python app/main.py
```

**Modo desenvolvimento** — suprime todos os writes em banco:

```bash
set IRIS_DEVMODE=1 && python app/main.py
```

<br/>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

## Semântica de Cores

| Cor | Hex | Significado |
|-----|-----|-------------|
| ![](https://placehold.co/14x14/FFB454/FFB454) Âmbar Óptico | `#FFB454` | Sistema ativo · interação · foco |
| ![](https://placehold.co/14x14/4ADE80/4ADE80) Fósforo Verde | `#4ADE80` | QR decodificado com sucesso |
| ![](https://placehold.co/14x14/FF7A7A/FF7A7A) Alerta | `#FF7A7A` | Fechamento · avisos |
| ![](https://placehold.co/14x14/8FA8BF/8FA8BF) Aço | `#8FA8BF` | Texto secundário · labels técnicos |
| ![](https://placehold.co/14x14/0B0E13/0B0E13) Grafite Profundo | `#0B0E13` | Fundo principal |
| ![](https://placehold.co/14x14/EAF0F6/EAF0F6) Off-White | `#EAF0F6` | Texto principal |

<br/>

<p align="center">
  <img src="docs/divider.svg" width="100%" alt=""/>
</p>

<br/>

<p align="center">
  <sub>
    <code>IRIS · SYSTEMS</code> &nbsp;·&nbsp; Estação de Leitura Óptica &nbsp;·&nbsp; PyQt6 · OpenCV · ONNX
  </sub>
</p>
