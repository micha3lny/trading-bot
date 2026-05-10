# Raspberry Pi live deployment

## Services

- ibgateway.service
- trading-bot.service
- portfolio-monitor.service

## Suggested directories

```bash
/home/pi/trading-bot
/home/pi/ibc
/home/pi/Jts
```

## Enable services

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ibgateway
sudo systemctl enable trading-bot
sudo systemctl enable portfolio-monitor
```

## Start

```bash
sudo systemctl start ibgateway
sudo systemctl start trading-bot
sudo systemctl start portfolio-monitor
```

## Logs

```bash
journalctl -u ibgateway -f
journalctl -u trading-bot -f
journalctl -u portfolio-monitor -f
```

## Restart

```bash
sudo systemctl restart trading-bot
```

## Recommended stack

- Raspberry Pi 5
- Raspberry Pi OS Lite
- Xvfb
- IBC + IB Gateway
- Python venv
- Recorder enabled
- Daily rsync backup to Mac
