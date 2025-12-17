# MarioMind

Projeto de Reinforcement Learning (RL) para treinar agentes jogando **Super Mario Bros (NES)**.  
O foco aqui é manter um repositório **profissional e reprodutível**, separando **código-fonte** de **artefatos gerados** (pesos, logs, scores, mídias).

Este repositório contém implementações próprias de:
- **Dueling DQN**
- **PPO (Actor-Critic)**

---

## PPO vs Dueling DQN (diferença e por que existem os dois)

Este projeto mantém **dois algoritmos** porque eles atacam o mesmo problema (aprender a jogar) com abordagens bem diferentes, o que ajuda tanto em estudo quanto em comparação prática:

- **Dueling DQN (value-based)** aprende uma função **Q(s, a)**: “qual ação tende a dar mais retorno nesse estado?”.  
  Em geral é ótimo para entender aprendizado por valores, replay buffer e target network. Costuma ser mais sensível a estabilidade/hiperparâmetros, mas é excelente para aprendizado e experimentos clássicos de RL em ambientes discretos.

- **PPO (policy-based / Actor-Critic)** aprende diretamente uma **política π(a|s)** (o “ator”) e estima valores (o “crítico”) para estabilizar o treino.  
  É um método moderno e geralmente mais estável em muitos cenários, usando atualizações “limitadas” (clipping) para evitar passos grandes demais. É muito usado como baseline robusto para tarefas mais complexas.

Na prática: **DQN** tende a ser um ótimo ponto de partida didático e base sólida para discretos; **PPO** costuma ser uma escolha forte quando você quer um treino mais estável e comparações com um algoritmo amplamente adotado.

---

## Objetivos da organização

- Manter a raiz do projeto **limpa** (sem `.pth/.pt/.p`, gifs, logs espalhados).
- Garantir que tudo que é gerado vá para `runs/` (ignorado no Git).
- Evitar “pastas vazias no Git” criando-as automaticamente em runtime.
- Padronizar execução com **um ponto único**: `main.py`.

---

## Estrutura do projeto

> Resumo do que você deve esperar após a organização.

```text
MarioMind/
  main.py
  requirements.txt
  README.md
  .gitignore

  mariomind/                 # código fonte
    env/                     # wrappers e ambiente
    algos/                   # ppo, duel_dqn
    eval/                    # execução/render do agente
    utils/                   # paths, criação de pastas, helpers

  assets/
    checkpoints/             # (opcional) checkpoints base versionados

  docs/
    assets/                  # gifs/imagens usadas no README

  runs/                      # tudo que for gerado (IGNORADO)
````

---

## O que vai para o Git e o que não vai

### Deve ser versionado (commitado)

* Código: `mariomind/`, `main.py`
* Configuração: `requirements.txt`, `.gitignore`
* Documentação: `README.md`
* Mídias de documentação: `docs/assets/` (ex.: gifs para mostrar resultado)
* (Opcional) checkpoints “base” que você realmente quer manter: `assets/checkpoints/`

### Não deve ser versionado

* Ambientes virtuais: `.venv/`, `venv/`
* Cache do Python: `__pycache__/`
* Artefatos de treino: `runs/`, logs, vídeos, scores
* Pesos e dumps gerados: `*.pt`, `*.pth`, `*.p` (quando não forem “base”)

---

## Por que criar a venv como `.venv` (e por que o VS Code recomenda)

`.venv` é apenas um nome, mas ele virou um padrão prático porque:

1. **Fica explícito** que a venv pertence a esse projeto (por estar dentro do repo).
2. O **VS Code detecta automaticamente** `.venv` com mais facilidade para selecionar o interpretador Python correto.
3. Evita confusões com ambientes globais e projetos diferentes.

Na prática, `.venv` e `venv` funcionam igual. Aqui padronizamos `.venv` e **não versionamos** essa pasta.

---

## Requisitos

* Windows 10/11 ou Linux/macOS
* Python **3.10.x** (recomendado para compatibilidade com bibliotecas do ecossistema)
* Dependências do `requirements.txt`

> Observação: projetos com Gym “legado” podem quebrar com NumPy 2.x.
> Por isso é comum manter `numpy<2` no requirements (quando aplicável ao seu stack).

---

## Instalação

### 1) Criar e ativar a venv

#### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Execução (um único ponto de entrada)

O projeto é executado pelo `main.py`, que centraliza as ações:

### Inicializar estrutura de pastas

Cria/garante diretórios como `runs/`, `assets/checkpoints/` e `docs/assets/` automaticamente (sem `gitkeep`).

```bash
python main.py init
```

### Treinar PPO

```bash
python main.py train ppo
```

### Treinar Dueling DQN

```bash
python main.py train dqn
```

### Assistir o agente (render)

```bash
python main.py play dqn --ckpt runs/<run_id>/models/mario_q_target.pth
```

Se você não informar `--ckpt`, o projeto tenta localizar um checkpoint “padrão” em locais esperados (ex.: `assets/checkpoints/`) conforme a estratégia definida internamente.

---

## Onde ficam os arquivos gerados

Cada execução cria uma pasta em `runs/` com layout padrão:

```text
runs/<algo>_YYYYmmdd_HHMMSS/
  models/     # checkpoints (.pt/.pth)
  logs/       # score.p, logs, métricas
  media/      # vídeos, frames, etc (se houver)
```

Isso mantém o repositório limpo e facilita comparar experimentos.

---

## Notas de arquitetura (importante)

* A **lógica dos algoritmos** (PPO/DQN) não é alterada por organização: apenas caminhos e persistência de arquivos foram padronizados.
* A camada `utils/` centraliza:

  * criação de diretórios
  * resolução de caminhos de checkpoints
  * padronização do destino dos artefatos
* A camada `env/` isola wrappers e configuração do ambiente.

---

## Troubleshooting rápido

* **`ModuleNotFoundError`**: confirme que a venv está ativa e rode `pip install -r requirements.txt`.
* **Erro Gym / NumPy**: garanta que seu ambiente está respeitando versões compatíveis (ex.: `numpy<2` quando necessário).
* **Sem render/janela**: alguns ambientes precisam de dependências de renderização (varia por SO), e o render pode ser pesado no CPU.
