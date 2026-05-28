# Raspberry Pi live deployment

## Services

- ibgateway.service
- trading-bot.service (production host may install/run this as `v67-trader.service`)
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
journalctl -u v67-trader -f
journalctl -u portfolio-monitor -f
```

The bot also writes unified daily logs to:

```bash
tail -f ~/trading-bot/data/logs/trading-bot-$(date -u +%F).log
```

## Restart

```bash
sudo systemctl restart v67-trader
```

## Shutdown / OOM diagnostics

If the bot exits cleanly but unexpectedly during market hours, check both the bot log and systemd/kernel state:

```bash
sudo systemctl status v67-trader --no-pager
journalctl -u v67-trader --since "2026-05-28 18:00" --no-pager
journalctl -k --since "2026-05-28 18:00" --no-pager | grep -Ei 'oom|out of memory|killed process|segfault|watchdog|thermal|under-voltage|voltage'
dmesg -T | grep -Ei 'oom|out of memory|killed process|segfault|watchdog|thermal|under-voltage|voltage'
```

Expected bot-side shutdown markers:

- `BOT_SIGNAL_RECEIVED` for `SIGTERM`, `SIGINT`, or `SIGHUP`
- `MAIN_LOOP_EXIT` when the main loop returns normally
- `IBKR_DISCONNECT_SOURCE` before intentional shutdown/reconnect disconnects
- `BOT_EXIT` / `BOT_STOP` from the final shutdown handler
- `UNEXPECTED_CLEAN_EXIT_DURING_SESSION` if exit code is 0 while the US equity session is open

## Recommended stack

- Raspberry Pi 5
- Raspberry Pi OS Lite
- Xvfb
- IBC + IB Gateway
- Python venv
- Recorder enabled
- Daily rsync backup to Mac
