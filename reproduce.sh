#!/bin/bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## reproduce
##

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NOCOLOR='\033[0m'

OVERRIDE_FLAG="--override-data"

set -e

echo -e "${BLUE}START${NOCOLOR} ${GREEN}TREK${NOCOLOR} - One-Click Reproduction"
echo ""

echo -e "${BLUE}[1/5] Cleaning old data.${NOCOLOR}"
echo ""

rm -rf data_qlearn data_dqn 2>/dev/null || true

echo -e "${GREEN}Cleaned${NOCOLOR}"
echo ""

echo -e "${BLUE}[2/5] Training Q-Learning.${NOCOLOR}"
python3 startTrek.py configs/qlearn.yaml $OVERRIDE_FLAG
echo ""
echo -e "${GREEN}Q-Learning training complete${NOCOLOR}"
echo ""

echo -e "${BLUE}[3/5] Training DQN.${NOCOLOR}"
python3 startTrek.py configs/dqn.yaml $OVERRIDE_FLAG
echo ""
echo -e "${GREEN}DQN training complete${NOCOLOR}"
echo ""

echo -e "${BLUE}[4/5] Eval Random.${NOCOLOR}"
python3 startTrek.py configs/random.yaml $OVERRIDE_FLAG
echo ""
echo -e "${GREEN}Random evaluation complete${NOCOLOR}"
echo ""

echo -e "${BLUE}[5/5] Generating reports.${NOCOLOR}"
if [ -f "review.py" ]; then
    ./review.py data_dqn
    ./review.py data_qlearn
    ./review.py data_baseline
    echo -e "${GREEN}Reports generated${NOCOLOR}"
else
    echo -e "${YELLOW}review.py not found${NOCOLOR}"
fi
echo ""

echo -e "${GREEN}Reproduction Complete.${NOCOLOR}"
echo ""
echo -e "${BLUE}Generated files:${NOCOLOR}"
echo -e "  ${GREEN}CSV:${NOCOLOR}"
echo -e "      - Q-Learning: data_qlearn/${CYAN}csv${NOCOLOR}/episodes.csv"
echo -e "      - DQN: data_dqn/${CYAN}csv${NOCOLOR}/episodes.csv"
echo -e "      - Random: data_baseline/${CYAN}csv${NOCOLOR}/episodes.csv"
echo -e "  ${GREEN}Videos:${NOCOLOR}"
echo -e "      - Q-Learning: data_qlearn/${CYAN}videos${NOCOLOR}/"
echo -e "      - DQN: data_dqn/${CYAN}videos${NOCOLOR}/"
echo -e "      - Random: data_baseline/${CYAN}videos${NOCOLOR}/"
echo ""
