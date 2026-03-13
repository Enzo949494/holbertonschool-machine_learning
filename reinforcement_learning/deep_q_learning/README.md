# Deep Q Learning - Breakout

Agent DQN pour jouer à Atari Breakout en utilisant le Reinforcement Learning.

## Installation

```bash
pip install -r requirements.txt
```

## Fichiers

- **train.py** - Entraîne le modèle DQN sur Breakout
- **play.py** - Lance le modèle entrainé pour voir le bot jouer (100 parties)
- **policy.h5** - Modèle entrainé sauvegardé

## Utilisation

### Entraîner le modèle
```bash
python train.py
```

### Voir le bot jouer
```bash
python play.py
```

## Notes

- Le modèle est entrainé avec Double DQN et atteint des scores ~30k
- Au test, `eps=0.0` pour une stratégie 100% greedy (meilleur score)
- La fenêtre Atari s'affiche en temps réel avec `render_mode="human"`
