"""
Advanced Features Module for Lottery Predictor (v1.5)
Portfolio optimization, syndicate/group play tools, notification system, multi-language support.
"""

import json
import smtplib
import ssl
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itertools import combinations

from .data import get_data_manager, LOTTERY_CONFIGS
from .models import EnsemblePredictor, PredictionResult, get_predictor


@dataclass
class PortfolioTicket:
    """A single ticket in a portfolio."""
    main_numbers: List[int]
    bonus_numbers: List[int]
    method: str
    confidence: float
    cost: float = 2.0  # Default $2 per ticket


@dataclass
class Portfolio:
    """A portfolio of lottery tickets."""
    lottery_type: str
    tickets: List[PortfolioTicket]
    total_cost: float
    budget: float
    strategy: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SyndicateMember:
    """A member of a lottery syndicate."""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    discord_id: Optional[str] = None
    share_percentage: float = 0.0
    contribution: float = 0.0


@dataclass
class Syndicate:
    """A lottery syndicate/group play."""
    name: str
    lottery_type: str
    members: List[SyndicateMember]
    tickets: List[PortfolioTicket]
    draw_date: Optional[str] = None
    total_cost: float = 0.0
    total_shares: float = 100.0


class PortfolioOptimizer:
    """Optimize lottery ticket portfolios for maximum coverage/diversity."""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS[lottery_type]

    def generate_diverse_portfolio(
        self,
        budget: float,
        ticket_cost: float = 2.0,
        method: str = "ensemble",
        diversity_weight: float = 0.5
    ) -> Portfolio:
        """Generate a diverse portfolio of tickets."""
        max_tickets = int(budget / ticket_cost)

        # Get predictions from multiple methods
        methods = ["statistical", "frequency", "cold", "pairs", "ml", "ensemble"]
        all_predictions = {}

        for m in methods:
            try:
                predictor = get_predictor(self.lottery_type, m)
                if m == "statistical":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_balanced(num_predictions=max_tickets * 2)
                elif m == "frequency":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_frequency_based(num_predictions=max_tickets * 2)
                elif m == "cold":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_cold_numbers(num_predictions=max_tickets * 2)
                elif m == "pairs":
                    pred = StatisticalPredictor(self.lottery_type)
                    preds = pred.predict_pair_based(num_predictions=max_tickets * 2)
                elif m == "ml":
                    pred = MLModelPredictor(self.lottery_type)
                    if not pred.is_trained:
                        pred.train()
                    preds = pred.predict(num_predictions=max_tickets * 2)
                else:  # ensemble
                    pred = EnsemblePredictor(self.lottery_type)
                    preds = pred.get_consensus_prediction(num_predictions=max_tickets * 2)

                all_predictions[m] = preds
            except Exception:
                continue

        # Select diverse tickets
        selected_tickets = []
        all_candidates = []

        for method, preds in all_predictions.items():
            for p in preds:
                all_candidates.append(PortfolioTicket(
                    main_numbers=p.main_numbers,
                    bonus_numbers=p.bonus_numbers,
                    method=method,
                    confidence=p.confidence,
                    cost=ticket_cost
                ))

        # Greedy diversity selection
        selected_tickets = self._select_diverse_tickets(
            all_candidates, max_tickets, diversity_weight
        )

        # If not enough, fill with random
        if len(selected_tickets) < max_tickets:
            from .data import LotteryDataManager
            dm = get_data_manager(self.lottery_type)
            df = dm.load_data()
            freq = dm.get_number_frequency(100)

            while len(selected_tickets) < max_tickets:
                main = np.random.choice(
                    range(1, self.config['main_max'] + 1),
                    size=self.config['main_numbers'],
                    replace=False
                )
                bonus = []
                if self.config['bonus_numbers'] > 0:
                    bonus = np.random.choice(
                        range(1, self.config['bonus_max'] + 1),
                        size=self.config['bonus_numbers'],
                        replace=False
                    ).tolist()

                ticket = PortfolioTicket(
                    main_numbers=sorted(main.tolist()),
                    bonus_numbers=bonus,
                    method="random",
                    confidence=0.1,
                    cost=ticket_cost
                )
                if not self._ticket_exists(ticket, selected_tickets):
                    selected_tickets.append(ticket)

        total_cost = sum(t.cost for t in selected_tickets)

        return Portfolio(
            lottery_type=self.lottery_type,
            tickets=selected_tickets,
            total_cost=total_cost,
            budget=budget,
            strategy="diverse_ensemble",
            metadata={"diversity_weight": diversity_weight, "methods_used": list(all_predictions.keys())}
        )

    def _select_diverse_tickets(
        self,
        candidates: List[PortfolioTicket],
        max_tickets: int,
        diversity_weight: float
    ) -> List[PortfolioTicket]:
        """Select tickets maximizing diversity."""
        if not candidates:
            return []

        # Sort by confidence * (1 - diversity_weight) + diversity_score * diversity_weight
        selected = [candidates[0]]
        remaining = candidates[1:]

        while len(selected) < max_tickets and remaining:
            best_idx = -1
            best_score = -1

            for i, candidate in enumerate(remaining):
                # Calculate diversity score (average distance to selected)
                diversity = self._calculate_diversity(candidate, selected)
                confidence_score = candidate.confidence
                combined = (confidence_score * (1 - diversity_weight) +
                           diversity * diversity_weight)

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    def _calculate_diversity(
        self,
        candidate: PortfolioTicket,
        selected: List[PortfolioTicket]
    ) -> float:
        """Calculate average Jaccard distance to selected tickets."""
        if not selected:
            return 1.0

        candidate_set = set(candidate.main_numbers + candidate.bonus_numbers)
        distances = []

        for t in selected:
            t_set = set(t.main_numbers + t.bonus_numbers)
            intersection = len(candidate_set & t_set)
            union = len(candidate_set | t_set)
            jaccard = intersection / union if union > 0 else 0
            distances.append(1 - jaccard)

        return np.mean(distances)

    def _ticket_exists(self, ticket: PortfolioTicket, tickets: List[PortfolioTicket]) -> bool:
        """Check if ticket already exists in list."""
        for t in tickets:
            if t.main_numbers == ticket.main_numbers and t.bonus_numbers == ticket.bonus_numbers:
                return True
        return False

    def optimize_wheel(self, numbers: List[int], ticket_cost: float = 2.0) -> Portfolio:
        """Generate wheel system (full or abbreviated) from a set of numbers."""
        main_count = self.config['main_numbers']

        if len(numbers) < main_count:
            raise ValueError(f"Need at least {main_count} numbers for wheel")

        # Generate combinations
        combos = list(combinations(sorted(numbers), main_count))

        tickets = []
        for combo in combos:
            bonus = []
            if self.config['bonus_numbers'] > 0:
                # Use remaining numbers for bonus
                remaining = [n for n in numbers if n not in combo]
                if remaining:
                    bonus = [remaining[0]]

            tickets.append(PortfolioTicket(
                main_numbers=list(combo),
                bonus_numbers=bonus,
                method="wheel",
                confidence=0.5,
                cost=ticket_cost
            ))

        total_cost = len(tickets) * ticket_cost
        return Portfolio(
            lottery_type=self.lottery_type,
            tickets=tickets,
            total_cost=total_cost,
            budget=total_cost,
            strategy=f"full_wheel_{len(numbers)}_numbers",
            metadata={"source_numbers": numbers}
        )


class SyndicateManager:
    """Manage lottery syndicates/group play."""

    def __init__(self):
        self.syndicates: Dict[str, Syndicate] = {}

    def create_syndicate(
        self,
        name: str,
        lottery_type: str,
        members: List[SyndicateMember],
        budget: float,
        ticket_cost: float = 2.0
    ) -> Syndicate:
        """Create a new syndicate with members and budget."""
        # Normalize shares
        total_shares = sum(m.share_percentage for m in members)
        if total_shares == 0:
            # Equal shares
            share = 100.0 / len(members)
            for m in members:
                m.share_percentage = share
                m.contribution = budget * share / 100.0
        else:
            for m in members:
                m.contribution = budget * m.share_percentage / 100.0

        # Generate tickets using portfolio optimizer
        optimizer = PortfolioOptimizer(lottery_type)
        portfolio = optimizer.generate_diverse_portfolio(budget, ticket_cost)

        syndicate = Syndicate(
            name=name,
            lottery_type=lottery_type,
            members=members,
            tickets=portfolio.tickets,
            total_cost=portfolio.total_cost,
            total_shares=100.0
        )

        self.syndicates[name] = syndicate
        return syndicate

    def get_syndicate(self, name: str) -> Optional[Syndicate]:
        """Get syndicate by name."""
        return self.syndicates.get(name)

    def distribute_winnings(self, syndicate_name: str, winnings: float) -> Dict[str, float]:
        """Distribute winnings among members based on shares."""
        syndicate = self.syndicates.get(syndicate_name)
        if not syndicate:
            return {}

        distributions = {}
        for member in syndicate.members:
            share = member.share_percentage / 100.0
            distributions[member.name] = winnings * share

        return distributions

    def export_syndicate(self, name: str, format: str = "json") -> str:
        """Export syndicate details."""
        syndicate = self.syndicates.get(name)
        if not syndicate:
            return ""

        data = {
            "name": syndicate.name,
            "lottery_type": syndicate.lottery_type,
            "draw_date": syndicate.draw_date,
            "total_cost": syndicate.total_cost,
            "total_shares": syndicate.total_shares,
            "members": [
                {
                    "name": m.name,
                    "email": m.email,
                    "phone": m.phone,
                    "discord_id": m.discord_id,
                    "share_percentage": m.share_percentage,
                    "contribution": m.contribution
                }
                for m in syndicate.members
            ],
            "tickets": [
                {
                    "main_numbers": t.main_numbers,
                    "bonus_numbers": t.bonus_numbers,
                    "method": t.method,
                    "confidence": t.confidence,
                    "cost": t.cost
                }
                for t in syndicate.tickets
            ]
        }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "csv":
            df = pd.DataFrame(data["tickets"])
            return df.to_csv(index=False)
        return ""


class NotificationSystem:
    """Notification system for lottery predictions and results."""

    def __init__(self):
        self.email_config: Optional[Dict] = None
        self.discord_webhook: Optional[str] = None
        self.pushbullet_token: Optional[str] = None

    def configure_email(self, smtp_server: str, port: int, username: str, password: str):
        """Configure email notifications."""
        self.email_config = {
            "smtp_server": smtp_server,
            "port": port,
            "username": username,
            "password": password
        }

    def configure_discord(self, webhook_url: str):
        """Configure Discord webhook."""
        self.discord_webhook = webhook_url

    def configure_pushbullet(self, token: str):
        """Configure Pushbullet for push notifications."""
        self.pushbullet_token = token

    def send_email(self, to_email: str, subject: str, body: str, html: bool = False) -> bool:
        """Send email notification."""
        if not self.email_config:
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config["username"]
            msg["To"] = to_email
            msg["Subject"] = subject

            if html:
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["port"]) as server:
                server.starttls(context=context)
                server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Email send failed: {e}")
            return False

    def send_discord(self, message: str, embed: Optional[Dict] = None) -> bool:
        """Send Discord notification."""
        if not self.discord_webhook:
            return False

        import requests
        try:
            payload = {"content": message}
            if embed:
                payload["embeds"] = [embed]

            response = requests.post(self.discord_webhook, json=payload)
            return response.status_code == 204
        except Exception as e:
            print(f"Discord send failed: {e}")
            return False

    def send_pushbullet(self, title: str, body: str) -> bool:
        """Send Pushbullet notification."""
        if not self.pushbullet_token:
            return False

        import requests
        try:
            headers = {
                "Access-Token": self.pushbullet_token,
                "Content-Type": "application/json"
            }
            payload = {
                "type": "note",
                "title": title,
                "body": body
            }
            response = requests.post("https://api.pushbullet.com/v2/pushes",
                                   headers=headers, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Pushbullet send failed: {e}")
            return False

    def notify_predictions(
        self,
        lottery_type: str,
        predictions: List[PredictionResult],
        recipients: List[str]
    ) -> Dict[str, bool]:
        """Send prediction notifications to multiple recipients."""
        config = LOTTERY_CONFIGS[lottery_type]

        # Format predictions
        pred_text = f"🎲 {config['name']} Predictions\n\n"
        for i, p in enumerate(predictions, 1):
            main = "  ".join(f"{n:2d}" for n in p.main_numbers)
            bonus = "  ".join(f"{n:2d}" for n in p.bonus_numbers) if p.bonus_numbers else "N/A"
            pred_text += f"{i}. {main}  +  {bonus}  (Confidence: {p.confidence:.1%}, Method: {p.method})\n"

        results = {}
        for recipient in recipients:
            if "@" in recipient:  # Email
                results[recipient] = self.send_email(
                    recipient,
                    f"🎲 {config['name']} Predictions for {datetime.now().strftime('%Y-%m-%d')}",
                    pred_text
                )
            elif recipient.startswith("discord:"):  # Discord
                webhook = recipient.replace("discord:", "")
                self.discord_webhook = webhook
                results[recipient] = self.send_discord(pred_text)
            elif recipient.startswith("pushbullet:"):  # Pushbullet
                token = recipient.replace("pushbullet:", "")
                self.pushbullet_token = token
                results[recipient] = self.send_pushbullet(
                    f"🎲 {config['name']} Predictions",
                    pred_text
                )

        return results


class I18nManager:
    """Multi-language support for the lottery predictor."""

    TRANSLATIONS = {
        "en": {
            "predictions": "Predictions",
            "main_numbers": "Main Numbers",
            "bonus_numbers": "Bonus Numbers",
            "confidence": "Confidence",
            "method": "Method",
            "disclaimer": "This tool is for entertainment and educational purposes only. Lottery draws are random events. No algorithm or AI can predict future lottery numbers with certainty. Past performance does not guarantee future results. Please play responsibly and within your means.",
            "hot_numbers": "Hot Numbers",
            "cold_numbers": "Cold Numbers",
            "frequency": "Frequency",
            "sum_statistics": "Sum Statistics",
            "pair_analysis": "Pair Analysis",
            "visualizations": "Visualizations",
            "history": "Recent Draws",
            "date": "Date",
            "jackpot": "Jackpot",
            "draw": "Draw",
            "export": "Export",
            "csv": "CSV",
            "excel": "Excel",
            "json": "JSON",
            "portfolio": "Portfolio",
            "syndicate": "Syndicate",
            "budget": "Budget",
            "tickets": "Tickets",
            "total_cost": "Total Cost",
            "winnings": "Winnings",
            "member": "Member",
            "share": "Share",
            "contribution": "Contribution",
            "optimize": "Optimize",
            "diversity": "Diversity",
            "wheel": "Wheel System",
            "notifications": "Notifications",
            "email": "Email",
            "discord": "Discord",
            "push": "Push Notification",
            "language": "Language",
            "settings": "Settings",
            "help": "Help",
            "about": "About",
            "version": "Version",
            "lottery_type": "Lottery Type",
            "prediction_method": "Prediction Method",
            "number_of_predictions": "Number of Predictions",
            "historical_draws": "Historical Draws to Analyze",
            "train_models": "Train ML Models",
            "generate_viz": "Generate Visualizations",
            "output_file": "Output File",
            "compare_methods": "Compare Methods",
            "validate_data": "Validate Data",
            "update_data": "Update Data",
            "schedule_updates": "Schedule Updates",
            "anomaly_detection": "Anomaly Detection",
        },
        "es": {
            "predictions": "Predicciones",
            "main_numbers": "Números Principales",
            "bonus_numbers": "Números Extra",
            "confidence": "Confianza",
            "method": "Método",
            "disclaimer": "Esta herramienta es solo para fines de entretenimiento y educativos. Los sorteos de lotería son eventos aleatorios. Ningún algoritmo o IA puede predecir números futuros con certeza. El rendimiento pasado no garantiza resultados futuros. Juegue responsablemente y dentro de sus posibilidades.",
            "hot_numbers": "Números Calientes",
            "cold_numbers": "Números Fríos",
            "frequency": "Frecuencia",
            "sum_statistics": "Estadísticas de Suma",
            "pair_analysis": "Análisis de Pares",
            "visualizations": "Visualizaciones",
            "history": "Sorteos Recientes",
            "date": "Fecha",
            "jackpot": "Bote",
            "draw": "Sorteo",
            "export": "Exportar",
            "csv": "CSV",
            "excel": "Excel",
            "json": "JSON",
            "portfolio": "Cartera",
            "syndicate": "Sindicato",
            "budget": "Presupuesto",
            "tickets": "Boletos",
            "total_cost": "Costo Total",
            "winnings": "Ganancias",
            "member": "Miembro",
            "share": "Participación",
            "contribution": "Aportación",
            "optimize": "Optimizar",
            "diversity": "Diversidad",
            "wheel": "Sistema de Rueda",
            "notifications": "Notificaciones",
            "email": "Correo Electrónico",
            "discord": "Discord",
            "push": "Notificación Push",
            "language": "Idioma",
            "settings": "Configuración",
            "help": "Ayuda",
            "about": "Acerca de",
            "version": "Versión",
            "lottery_type": "Tipo de Lotería",
            "prediction_method": "Método de Predicción",
            "number_of_predictions": "Número de Predicciones",
            "historical_draws": "Sorteos Históricos a Analizar",
            "train_models": "Entrenar Modelos ML",
            "generate_viz": "Generar Visualizaciones",
            "output_file": "Archivo de Salida",
            "compare_methods": "Comparar Métodos",
            "validate_data": "Validar Datos",
            "update_data": "Actualizar Datos",
            "schedule_updates": "Programar Actualizaciones",
            "anomaly_detection": "Detección de Anomalías",
        },
        "fr": {
            "predictions": "Prédictions",
            "main_numbers": "Numéros Principaux",
            "bonus_numbers": "Numéros Bonus",
            "confidence": "Confiance",
            "method": "Méthode",
            "disclaimer": "Cet outil est destiné uniquement à des fins de divertissement et éducatives. Les tirages de loterie sont des événements aléatoires. Aucun algorithme ou IA ne peut prédire les numéros futurs avec certitude. Les performances passées ne garantissent pas les résultats futurs. Jouez de manière responsable et dans vos moyens.",
            "hot_numbers": "Numéros Chauds",
            "cold_numbers": "Numéros Froids",
            "frequency": "Fréquence",
            "sum_statistics": "Statistiques de Somme",
            "pair_analysis": "Analyse de Paires",
            "visualizations": "Visualisations",
            "history": "Tirages Récents",
            "date": "Date",
            "jackpot": "Jackpot",
            "draw": "Tirage",
            "export": "Exporter",
            "csv": "CSV",
            "excel": "Excel",
            "json": "JSON",
            "portfolio": "Portefeuille",
            "syndicate": "Syndicat",
            "budget": "Budget",
            "tickets": "Billets",
            "total_cost": "Coût Total",
            "winnings": "Gains",
            "member": "Membre",
            "share": "Part",
            "contribution": "Contribution",
            "optimize": "Optimiser",
            "diversity": "Diversité",
            "wheel": "Système de Roue",
            "notifications": "Notifications",
            "email": "Email",
            "discord": "Discord",
            "push": "Notification Push",
            "language": "Langue",
            "settings": "Paramètres",
            "help": "Aide",
            "about": "À propos",
            "version": "Version",
            "lottery_type": "Type de Loterie",
            "prediction_method": "Méthode de Prédiction",
            "number_of_predictions": "Nombre de Prédictions",
            "historical_draws": "Tirages Historiques à Analyser",
            "train_models": "Entraîner Modèles ML",
            "generate_viz": "Générer Visualisations",
            "output_file": "Fichier de Sortie",
            "compare_methods": "Comparer Méthodes",
            "validate_data": "Valider Données",
            "update_data": "Mettre à Jour Données",
            "schedule_updates": "Planifier Mises à Jour",
            "anomaly_detection": "Détection d'Anomalies",
        },
        "de": {
            "predictions": "Vorhersagen",
            "main_numbers": "Hauptzahlen",
            "bonus_numbers": "Zusatzzahlen",
            "confidence": "Vertrauen",
            "method": "Methode",
            "disclaimer": "Dieses Tool dient nur Unterhaltungs- und Bildungszwecken. Lottoziehungen sind zufällige Ereignisse. Kein Algorithmus oder KI kann zukünftige Zahlen mit Sicherheit vorhersagen. Vergangene Leistung garantiert keine zukünftigen Ergebnisse. Spielen Sie verantwortungsbewusst und innerhalb Ihrer Möglichkeiten.",
            "hot_numbers": "Heiße Zahlen",
            "cold_numbers": "Kalte Zahlen",
            "frequency": "Häufigkeit",
            "sum_statistics": "Summenstatistik",
            "pair_analysis": "Paaranalyse",
            "visualizations": "Visualisierungen",
            "history": "Letzte Ziehungen",
            "date": "Datum",
            "jackpot": "Jackpot",
            "draw": "Ziehung",
            "export": "Exportieren",
            "csv": "CSV",
            "excel": "Excel",
            "json": "JSON",
            "portfolio": "Portfolio",
            "syndicate": "Syndikat",
            "budget": "Budget",
            "tickets": "Lose",
            "total_cost": "Gesamtkosten",
            "winnings": "Gewinne",
            "member": "Mitglied",
            "share": "Anteil",
            "contribution": "Beitrag",
            "optimize": "Optimieren",
            "diversity": "Vielfalt",
            "wheel": "Rad-System",
            "notifications": "Benachrichtigungen",
            "email": "E-Mail",
            "discord": "Discord",
            "push": "Push-Benachrichtigung",
            "language": "Sprache",
            "settings": "Einstellungen",
            "help": "Hilfe",
            "about": "Über",
            "version": "Version",
            "lottery_type": "Lotterietyp",
            "prediction_method": "Vorhersagemethode",
            "number_of_predictions": "Anzahl der Vorhersagen",
            "historical_draws": "Historische Ziehungen zur Analyse",
            "train_models": "ML-Modelle Trainieren",
            "generate_viz": "Visualisierungen Erzeugen",
            "output_file": "Ausgabedatei",
            "compare_methods": "Methoden Vergleichen",
            "validate_data": "Daten Validieren",
            "update_data": "Daten Aktualisieren",
            "schedule_updates": "Updates Planen",
            "anomaly_detection": "Anomalieerkennung",
        },
        "zh": {
            "predictions": "预测",
            "main_numbers": "主号码",
            "bonus_numbers": "特别号码",
            "confidence": "置信度",
            "method": "方法",
            "disclaimer": "本工具仅供娱乐和教育目的。彩票开奖是随机事件。没有算法或人工智能能确定地预测未来的彩票号码。过往表现不代表未来结果。请理性购彩，量力而行。",
            "hot_numbers": "热号",
            "cold_numbers": "冷号",
            "frequency": "频率",
            "sum_statistics": "和值统计",
            "pair_analysis": "组合分析",
            "visualizations": "可视化",
            "history": "历史开奖",
            "date": "日期",
            "jackpot": "头奖",
            "draw": "期数",
            "export": "导出",
            "csv": "CSV",
            "excel": "Excel",
            "json": "JSON",
            "portfolio": "投资组合",
            "syndicate": "合买",
            "budget": "预算",
            "tickets": "注数",
            "total_cost": "总成本",
            "winnings": "奖金",
            "member": "成员",
            "share": "份额",
            "contribution": "出资",
            "optimize": "优化",
            "diversity": "多样性",
            "wheel": "轮盘系统",
            "notifications": "通知",
            "email": "邮件",
            "discord": "Discord",
            "push": "推送通知",
            "language": "语言",
            "settings": "设置",
            "help": "帮助",
            "about": "关于",
            "version": "版本",
            "lottery_type": "彩票类型",
            "prediction_method": "预测方法",
            "number_of_predictions": "预测数量",
            "historical_draws": "分析历史期数",
            "train_models": "训练ML模型",
            "generate_viz": "生成可视化",
            "output_file": "输出文件",
            "compare_methods": "对比方法",
            "validate_data": "验证数据",
            "update_data": "更新数据",
            "schedule_updates": "定时更新",
            "anomaly_detection": "异常检测",
        },
    }

    def __init__(self, language: str = "en"):
        self.language = language if language in self.TRANSLATIONS else "en"

    def set_language(self, language: str):
        """Set current language."""
        if language in self.TRANSLATIONS:
            self.language = language

    def get(self, key: str) -> str:
        """Get translation for key."""
        return self.TRANSLATIONS.get(self.language, {}).get(key, key)

    def get_all(self) -> Dict[str, str]:
        """Get all translations for current language."""
        return self.TRANSLATIONS.get(self.language, {})

    @classmethod
    def get_available_languages(cls) -> List[str]:
        """Get list of available languages."""
        return list(cls.TRANSLATIONS.keys())

    @classmethod
    def add_language(cls, code: str, translations: Dict[str, str]):
        """Add a new language."""
        cls.TRANSLATIONS[code] = translations


# Import here to avoid circular imports
from .models import StatisticalPredictor, MLModelPredictor


# ============================================================================
# v2.2 - User Experience & Social: User Profiles & Preferences
# ============================================================================

@dataclass
class UserPreferences:
    """User preferences for lottery predictor."""
    favorite_lotteries: List[str] = field(default_factory=list)
    default_lottery: str = "powerball"
    default_method: str = "ensemble"
    default_prediction_count: int = 5
    default_history_depth: int = 200
    language: str = "en"
    theme: str = "system"  # light, dark, system
    notifications: Dict[str, Any] = field(default_factory=lambda: {
        "email_enabled": False,
        "email_address": None,
        "discord_enabled": False,
        "discord_webhook": None,
        "pushbullet_enabled": False,
        "pushbullet_token": None,
        "smtp_server": None,
        "smtp_port": 587,
        "smtp_user": None,
        "smtp_pass": None,
    })
    auto_detect_state: bool = True
    use_all_history: bool = False


@dataclass
class UserProfile:
    """User profile with preferences and history."""
    username: str
    email: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None
    preferences: UserPreferences = field(default_factory=UserPreferences)
    prediction_history: List[Dict] = field(default_factory=list)
    syndicate_memberships: List[str] = field(default_factory=list)
    total_spent: float = 0.0
    total_won: float = 0.0

    def add_prediction_record(self, lottery: str, method: str, predictions: List[Dict]):
        """Add a prediction to history."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "lottery": lottery,
            "method": method,
            "predictions": predictions
        }
        self.prediction_history.append(record)
        # Keep only last 1000 records
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def get_favorite_lotteries(self) -> List[str]:
        """Get user's favorite lotteries, fall back to defaults."""
        if self.preferences.favorite_lotteries:
            return self.preferences.favorite_lotteries
        return ["powerball", "mega_millions", "florida_lotto"]


class UserProfileManager:
    """Manage user profiles and preferences."""

    def __init__(self, storage_path: str = "user_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.current_user_file = self.storage_path / ".current_user"
        self.profiles: Dict[str, UserProfile] = {}
        self.current_user: Optional[str] = None
        self._load_all_profiles()
        self._load_current_user()

    def _get_profile_path(self, username: str) -> Path:
        """Get file path for user profile."""
        safe_name = "".join(c for c in username if c.isalnum() or c in "-_")
        return self.storage_path / f"{safe_name}.json"

    def _load_all_profiles(self):
        """Load all profiles from disk."""
        for profile_file in self.storage_path.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                profile = self._deserialize_profile(data)
                self.profiles[profile.username] = profile
            except Exception as e:
                print(f"Warning: Failed to load profile {profile_file}: {e}")

    def _load_current_user(self):
        """Load current user from disk."""
        if self.current_user_file.exists():
            try:
                with open(self.current_user_file, 'r') as f:
                    self.current_user = f.read().strip()
            except Exception:
                pass

    def _save_current_user(self):
        """Save current user to disk."""
        try:
            with open(self.current_user_file, 'w') as f:
                f.write(self.current_user or "")
        except Exception:
            pass

    def _deserialize_profile(self, data: Dict) -> UserProfile:
        """Deserialize profile from dict."""
        prefs_data = data.get("preferences", {})
        prefs = UserPreferences(
            favorite_lotteries=prefs_data.get("favorite_lotteries", []),
            default_lottery=prefs_data.get("default_lottery", "powerball"),
            default_method=prefs_data.get("default_method", "ensemble"),
            default_prediction_count=prefs_data.get("default_prediction_count", 5),
            default_history_depth=prefs_data.get("default_history_depth", 200),
            language=prefs_data.get("language", "en"),
            theme=prefs_data.get("theme", "system"),
            notifications=prefs_data.get("notifications", UserPreferences().notifications),
            auto_detect_state=prefs_data.get("auto_detect_state", True),
            use_all_history=prefs_data.get("use_all_history", False),
        )

        profile = UserProfile(
            username=data["username"],
            email=data.get("email"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_login=data.get("last_login"),
            preferences=prefs,
            prediction_history=data.get("prediction_history", []),
            syndicate_memberships=data.get("syndicate_memberships", []),
            total_spent=data.get("total_spent", 0.0),
            total_won=data.get("total_won", 0.0),
        )
        return profile

    def _serialize_profile(self, profile: UserProfile) -> Dict:
        """Serialize profile to dict."""
        return {
            "username": profile.username,
            "email": profile.email,
            "created_at": profile.created_at,
            "last_login": profile.last_login,
            "preferences": {
                "favorite_lotteries": profile.preferences.favorite_lotteries,
                "default_lottery": profile.preferences.default_lottery,
                "default_method": profile.preferences.default_method,
                "default_prediction_count": profile.preferences.default_prediction_count,
                "default_history_depth": profile.preferences.default_history_depth,
                "language": profile.preferences.language,
                "theme": profile.preferences.theme,
                "notifications": profile.preferences.notifications,
                "auto_detect_state": profile.preferences.auto_detect_state,
                "use_all_history": profile.preferences.use_all_history,
            },
            "prediction_history": profile.prediction_history,
            "syndicate_memberships": profile.syndicate_memberships,
            "total_spent": profile.total_spent,
            "total_won": profile.total_won,
        }

    def create_profile(self, username: str, email: Optional[str] = None) -> UserProfile:
        """Create a new user profile."""
        if username in self.profiles:
            raise ValueError(f"Profile '{username}' already exists")

        profile = UserProfile(username=username, email=email)
        self.profiles[username] = profile
        self._save_profile(profile)
        return profile

    def get_profile(self, username: str) -> Optional[UserProfile]:
        """Get user profile by username."""
        return self.profiles.get(username)

    def get_or_create_profile(self, username: str, email: Optional[str] = None) -> UserProfile:
        """Get existing profile or create new one."""
        profile = self.profiles.get(username)
        if not profile:
            profile = self.create_profile(username, email)
        return profile

    def set_current_user(self, username: str) -> bool:
        """Set current active user."""
        if username in self.profiles:
            self.current_user = username
            self._save_current_user()
            # Update last login
            self.profiles[username].last_login = datetime.now().isoformat()
            self._save_profile(self.profiles[username])
            return True
        return False

    def get_current_profile(self) -> Optional[UserProfile]:
        """Get current active user profile."""
        if self.current_user:
            return self.profiles.get(self.current_user)
        return None

    def _save_profile(self, profile: UserProfile):
        """Save profile to disk."""
        profile_path = self._get_profile_path(profile.username)
        with open(profile_path, 'w') as f:
            json.dump(self._serialize_profile(profile), f, indent=2)

    def update_preferences(self, username: str, **kwargs) -> bool:
        """Update user preferences."""
        profile = self.profiles.get(username)
        if not profile:
            return False

        prefs = profile.preferences
        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        self._save_profile(profile)
        return True

    def add_favorite_lottery(self, username: str, lottery: str) -> bool:
        """Add lottery to user's favorites."""
        profile = self.profiles.get(username)
        if not profile:
            return False

        if lottery not in profile.preferences.favorite_lotteries:
            profile.preferences.favorite_lotteries.append(lottery)
            self._save_profile(profile)
        return True

    def remove_favorite_lottery(self, username: str, lottery: str) -> bool:
        """Remove lottery from user's favorites."""
        profile = self.profiles.get(username)
        if not profile:
            return False

        if lottery in profile.preferences.favorite_lotteries:
            profile.preferences.favorite_lotteries.remove(lottery)
            self._save_profile(profile)
        return True

    def add_syndicate_membership(self, username: str, syndicate_name: str) -> bool:
        """Add syndicate membership to user."""
        profile = self.profiles.get(username)
        if not profile:
            return False

        if syndicate_name not in profile.syndicate_memberships:
            profile.syndicate_memberships.append(syndicate_name)
            self._save_profile(profile)
        return True

    def record_spending(self, username: str, amount: float):
        """Record spending for user."""
        profile = self.profiles.get(username)
        if profile:
            profile.total_spent += amount
            self._save_profile(profile)

    def record_winnings(self, username: str, amount: float):
        """Record winnings for user."""
        profile = self.profiles.get(username)
        if profile:
            profile.total_won += amount
            self._save_profile(profile)

    def get_roi(self, username: str) -> Optional[float]:
        """Calculate ROI for user."""
        profile = self.profiles.get(username)
        if not profile or profile.total_spent == 0:
            return None
        return ((profile.total_won - profile.total_spent) / profile.total_spent) * 100

    def list_profiles(self) -> List[str]:
        """List all usernames."""
        return list(self.profiles.keys())

    def delete_profile(self, username: str) -> bool:
        """Delete user profile."""
        if username not in self.profiles:
            return False

        profile_path = self._get_profile_path(username)
        if profile_path.exists():
            profile_path.unlink()

        del self.profiles[username]
        if self.current_user == username:
            self.current_user = None
        return True


# ============================================================================
# Social Features: Prediction Sharing & Leaderboards
# ============================================================================

@dataclass
class SharedPrediction:
    """A shared prediction for community features."""
    id: str
    username: str
    lottery: str
    method: str
    predictions: List[Dict]
    timestamp: str
    is_anonymous: bool = False
    likes: int = 0
    comments: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "username": self.username,
            "lottery": self.lottery,
            "method": self.method,
            "predictions": self.predictions,
            "timestamp": self.timestamp,
            "is_anonymous": self.is_anonymous,
            "likes": self.likes,
            "comments": self.comments
        }


class SocialManager:
    """Manage social features: sharing, leaderboards, community pools."""

    def __init__(self, storage_path: str = "social_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.shared_predictions: List[SharedPrediction] = []
        self.leaderboard: Dict[str, Dict] = {}  # username -> stats
        self._load_data()

    def _load_data(self):
        """Load social data from disk."""
        shared_file = self.storage_path / "shared_predictions.json"
        if shared_file.exists():
            try:
                with open(shared_file, 'r') as f:
                    data = json.load(f)
                self.shared_predictions = [SharedPrediction(**p) for p in data]
            except Exception:
                pass

        leaderboard_file = self.storage_path / "leaderboard.json"
        if leaderboard_file.exists():
            try:
                with open(leaderboard_file, 'r') as f:
                    self.leaderboard = json.load(f)
            except Exception:
                pass

    def _save_shared_predictions(self):
        """Save shared predictions to disk."""
        shared_file = self.storage_path / "shared_predictions.json"
        with open(shared_file, 'w') as f:
            json.dump([p.to_dict() for p in self.shared_predictions], f, indent=2)

    def _save_leaderboard(self):
        """Save leaderboard to disk."""
        leaderboard_file = self.storage_path / "leaderboard.json"
        with open(leaderboard_file, 'w') as f:
            json.dump(self.leaderboard, f, indent=2)

    def share_prediction(
        self,
        username: str,
        lottery: str,
        method: str,
        predictions: List[Dict],
        is_anonymous: bool = False
    ) -> SharedPrediction:
        """Share a prediction with the community."""
        import uuid
        prediction_id = str(uuid.uuid4())[:8]

        shared = SharedPrediction(
            id=prediction_id,
            username=username if not is_anonymous else "Anonymous",
            lottery=lottery,
            method=method,
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
            is_anonymous=is_anonymous
        )

        self.shared_predictions.append(shared)
        # Keep only last 500 shared predictions
        if len(self.shared_predictions) > 500:
            self.shared_predictions = self.shared_predictions[-500:]

        self._save_shared_predictions()
        return shared

    def like_prediction(self, prediction_id: str) -> bool:
        """Like a shared prediction."""
        for pred in self.shared_predictions:
            if pred.id == prediction_id:
                pred.likes += 1
                self._save_shared_predictions()
                return True
        return False

    def add_comment(self, prediction_id: str, username: str, comment: str) -> bool:
        """Add comment to a shared prediction."""
        for pred in self.shared_predictions:
            if pred.id == prediction_id:
                pred.comments.append({
                    "username": username,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                })
                self._save_shared_predictions()
                return True
        return False

    def get_shared_predictions(
        self,
        lottery: Optional[str] = None,
        limit: int = 20
    ) -> List[SharedPrediction]:
        """Get shared predictions, optionally filtered by lottery."""
        results = self.shared_predictions
        if lottery:
            results = [p for p in results if p.lottery == lottery]
        return results[-limit:]

    def get_leaderboard(self, lottery: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get accuracy leaderboard."""
        # Calculate accuracy stats for each user
        stats = []
        for username, data in self.leaderboard.items():
            total = data.get("total_predictions", 0)
            if total == 0:
                continue

            correct = data.get("correct_predictions", 0)
            accuracy = (correct / total * 100) if total > 0 else 0

            stats.append({
                "username": username,
                "total_predictions": total,
                "correct_predictions": correct,
                "accuracy": round(accuracy, 2),
                "lottery": lottery or "all"
            })

        # Sort by accuracy descending
        stats.sort(key=lambda x: x["accuracy"], reverse=True)
        return stats[:limit]

    def record_prediction_result(self, username: str, lottery: str, was_correct: bool):
        """Record prediction result for leaderboard."""
        if username not in self.leaderboard:
            self.leaderboard[username] = {
                "total_predictions": 0,
                "correct_predictions": 0
            }

        self.leaderboard[username]["total_predictions"] += 1
        if was_correct:
            self.leaderboard[username]["correct_predictions"] += 1

        self._save_leaderboard()

    def create_community_pool(
        self,
        name: str,
        lottery: str,
        budget: float,
        ticket_cost: float = 2.0,
        creator: str = "system"
    ) -> Dict:
        """Create a community prediction pool."""
        import uuid
        pool_id = str(uuid.uuid4())[:8]

        pool = {
            "id": pool_id,
            "name": name,
            "lottery": lottery,
            "budget": budget,
            "ticket_cost": ticket_cost,
            "creator": creator,
            "participants": [],
            "tickets": [],
            "created_at": datetime.now().isoformat(),
            "status": "open"
        }

        pool_file = self.storage_path / f"pool_{pool_id}.json"
        with open(pool_file, 'w') as f:
            json.dump(pool, f, indent=2)

        return pool

    def join_pool(self, pool_id: str, username: str, contribution: float) -> bool:
        """Join a community pool."""
        pool_file = self.storage_path / f"pool_{pool_id}.json"
        if not pool_file.exists():
            return False

        with open(pool_file, 'r') as f:
            pool = json.load(f)

        if pool["status"] != "open":
            return False

        # Check if user already in pool
        for p in pool["participants"]:
            if p["username"] == username:
                return False

        pool["participants"].append({
            "username": username,
            "contribution": contribution,
            "joined_at": datetime.now().isoformat()
        })

        with open(pool_file, 'w') as f:
            json.dump(pool, f, indent=2)

        return True

    def get_pool(self, pool_id: str) -> Optional[Dict]:
        """Get pool details."""
        pool_file = self.storage_path / f"pool_{pool_id}.json"
        if not pool_file.exists():
            return None

        with open(pool_file, 'r') as f:
            return json.load(f)

    def list_pools(self, status: Optional[str] = None) -> List[Dict]:
        """List all community pools."""
        pools = []
        for pool_file in self.storage_path.glob("pool_*.json"):
            try:
                with open(pool_file, 'r') as f:
                    pool = json.load(f)
                if status is None or pool.get("status") == status:
                    pools.append(pool)
            except Exception:
                pass
        return pools


# ============================================================================
# v2.3 - Intelligence & Automation: Model Performance, Budget Tracking, Multi-Draw
# ============================================================================

@dataclass
class ModelPerformanceRecord:
    """A record of model performance for a specific period."""
    timestamp: str
    lottery: str
    method: str
    accuracy: float
    hit_rate: float
    roi: float
    num_predictions: int
    num_draws_verified: int
    model_version: str = "1.0"


@dataclass
class BudgetRecord:
    """A budget/spending record."""
    timestamp: str
    lottery: str
    amount_spent: float
    amount_won: float
    tickets_purchased: int
    net: float


class ModelPerformanceDashboard:
    """Track model performance over time, detect drift, and visualize results."""

    def __init__(self, storage_path: str = "model_performance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.performance_file = self.storage_path / "performance_history.json"
        self.records: List[ModelPerformanceRecord] = []
        self._load_records()

    def _load_records(self):
        """Load performance records from disk."""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                self.records = [ModelPerformanceRecord(**r) for r in data]
            except Exception:
                pass

    def _save_records(self):
        """Save performance records to disk."""
        with open(self.performance_file, 'w') as f:
            json.dump([r.__dict__ for r in self.records], f, indent=2)

    def record_performance(
        self,
        lottery: str,
        method: str,
        accuracy: float,
        hit_rate: float,
        roi: float,
        num_predictions: int,
        num_draws_verified: int,
        model_version: str = "1.0"
    ) -> ModelPerformanceRecord:
        """Record a performance metric."""
        record = ModelPerformanceRecord(
            timestamp=datetime.now().isoformat(),
            lottery=lottery,
            method=method,
            accuracy=accuracy,
            hit_rate=hit_rate,
            roi=roi,
            num_predictions=num_predictions,
            num_draws_verified=num_draws_verified,
            model_version=model_version
        )
        self.records.append(record)
        self._save_records()
        return record

    def get_performance_history(
        self,
        lottery: Optional[str] = None,
        method: Optional[str] = None,
        days: int = 30
    ) -> List[ModelPerformanceRecord]:
        """Get performance history filtered by lottery/method and time."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)

        results = []
        for r in self.records:
            record_time = datetime.fromisoformat(r.timestamp)
            if record_time < cutoff:
                continue
            if lottery and r.lottery != lottery:
                continue
            if method and r.method != method:
                continue
            results.append(r)

        return results

    def get_accuracy_trend(
        self,
        lottery: str,
        method: str,
        days: int = 30
    ) -> List[Dict]:
        """Get accuracy trend over time."""
        records = self.get_performance_history(lottery, method, days)
        records.sort(key=lambda x: x.timestamp)

        return [
            {
                "timestamp": r.timestamp[:10],
                "accuracy": r.accuracy,
                "hit_rate": r.hit_rate,
                "roi": r.roi,
                "predictions": r.num_predictions
            }
            for r in records
        ]

    def detect_drift(
        self,
        lottery: str,
        method: str,
        threshold: float = 0.05,
        days: int = 30
    ) -> Dict:
        """Detect model performance drift."""
        records = self.get_performance_history(lottery, method, days)
        if len(records) < 3:
            return {"drift_detected": False, "message": "Insufficient data"}

        records.sort(key=lambda x: x.timestamp)
        accuracies = [r.accuracy for r in records]

        # Compare first half vs second half
        mid = len(accuracies) // 2
        first_half_avg = sum(accuracies[:mid]) / mid
        second_half_avg = sum(accuracies[mid:]) / (len(accuracies) - mid)

        drift = second_half_avg - first_half_avg
        drift_detected = abs(drift) > threshold

        return {
            "drift_detected": drift_detected,
            "drift_magnitude": drift,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
            "threshold": threshold,
            "samples": len(records)
        }

    def get_method_comparison(self, lottery: str, days: int = 30) -> List[Dict]:
        """Compare all methods for a lottery."""
        methods = ["statistical", "frequency", "cold", "pairs", "ml", "ensemble", "time_series"]
        results = []

        for method in methods:
            records = self.get_performance_history(lottery, method, days)
            if not records:
                continue

            avg_accuracy = sum(r.accuracy for r in records) / len(records)
            avg_hit_rate = sum(r.hit_rate for r in records) / len(records)
            avg_roi = sum(r.roi for r in records) / len(records)
            total_predictions = sum(r.num_predictions for r in records)

            results.append({
                "method": method,
                "avg_accuracy": avg_accuracy,
                "avg_hit_rate": avg_hit_rate,
                "avg_roi": avg_roi,
                "total_predictions": total_predictions,
                "periods": len(records)
            })

        results.sort(key=lambda x: x["avg_accuracy"], reverse=True)
        return results

    def should_retrain(self, lottery: str, method: str, threshold: float = 0.05) -> bool:
        """Check if model should be retrained based on performance drift."""
        drift = self.detect_drift(lottery, method, threshold)
        return drift["drift_detected"] and drift["drift_magnitude"] < 0


class BudgetTracker:
    """Track spending vs winnings over time with per-lottery reports and tax estimation."""

    def __init__(self, storage_path: str = "budget_tracker"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.budget_file = self.storage_path / "budget_history.json"
        self.records: List[BudgetRecord] = []
        self._load_records()

    def _load_records(self):
        """Load budget records from disk."""
        if self.budget_file.exists():
            try:
                with open(self.budget_file, 'r') as f:
                    data = json.load(f)
                self.records = [BudgetRecord(**r) for r in data]
            except Exception:
                pass

    def _save_records(self):
        """Save budget records to disk."""
        with open(self.budget_file, 'w') as f:
            json.dump([r.__dict__ for r in self.records], f, indent=2)

    def record_spending(
        self,
        lottery: str,
        amount_spent: float,
        amount_won: float = 0.0,
        tickets_purchased: int = 1
    ) -> BudgetRecord:
        """Record a spending/winning entry."""
        net = amount_won - amount_spent
        record = BudgetRecord(
            timestamp=datetime.now().isoformat(),
            lottery=lottery,
            amount_spent=amount_spent,
            amount_won=amount_won,
            tickets_purchased=tickets_purchased,
            net=net
        )
        self.records.append(record)
        self._save_records()
        return record

    def get_summary(
        self,
        lottery: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict:
        """Get budget summary."""
        from datetime import timedelta

        records = self.records

        if lottery:
            records = [r for r in records if r.lottery == lottery]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            records = [r for r in records if datetime.fromisoformat(r.timestamp) >= cutoff]

        if not records:
            return {
                "total_spent": 0.0,
                "total_won": 0.0,
                "net": 0.0,
                "total_tickets": 0,
                "roi": 0.0,
                "records": 0
            }

        total_spent = sum(r.amount_spent for r in records)
        total_won = sum(r.amount_won for r in records)
        total_tickets = sum(r.tickets_purchased for r in records)
        net = total_won - total_spent
        roi = (net / total_spent * 100) if total_spent > 0 else 0.0

        return {
            "total_spent": total_spent,
            "total_won": total_won,
            "net": net,
            "total_tickets": total_tickets,
            "roi": roi,
            "records": len(records)
        }

    def get_per_lottery_breakdown(self, days: Optional[int] = None) -> List[Dict]:
        """Get spending breakdown per lottery."""
        lotteries = set(r.lottery for r in self.records)
        results = []

        for lottery in lotteries:
            summary = self.get_summary(lottery, days)
            summary["lottery"] = lottery
            results.append(summary)

        results.sort(key=lambda x: x["total_spent"], reverse=True)
        return results

    def get_monthly_trend(self, months: int = 12) -> List[Dict]:
        """Get monthly spending/winnings trend."""
        from collections import defaultdict

        monthly = defaultdict(lambda: {"spent": 0.0, "won": 0.0, "tickets": 0})

        for r in self.records:
            month_key = r.timestamp[:7]  # YYYY-MM
            monthly[month_key]["spent"] += r.amount_spent
            monthly[month_key]["won"] += r.amount_won
            monthly[month_key]["tickets"] += r.tickets_purchased

        # Sort and take last N months
        sorted_months = sorted(monthly.items())[-months:]

        return [
            {
                "month": month,
                "spent": data["spent"],
                "won": data["won"],
                "net": data["won"] - data["spent"],
                "tickets": data["tickets"],
                "roi": ((data["won"] - data["spent"]) / data["spent"] * 100) if data["spent"] > 0 else 0.0
            }
            for month, data in sorted_months
        ]

    def estimate_tax(
        self,
        winnings: float,
        federal_rate: float = 0.24,
        state_rate: float = 0.0,
        threshold: float = 600.0
    ) -> Dict:
        """Estimate tax on lottery winnings."""
        # Winnings over $600 are typically reported
        if winnings < threshold:
            return {
                "taxable": False,
                "federal_tax": 0.0,
                "state_tax": 0.0,
                "total_tax": 0.0,
                "net_after_tax": winnings
            }

        federal_tax = winnings * federal_rate
        state_tax = winnings * state_rate
        total_tax = federal_tax + state_tax

        return {
            "taxable": True,
            "federal_tax": federal_tax,
            "state_tax": state_tax,
            "total_tax": total_tax,
            "net_after_tax": winnings - total_tax,
            "federal_rate": federal_rate,
            "state_rate": state_rate
        }

    def export_report(
        self,
        format: str = "json",
        output_path: Optional[str] = None,
        days: Optional[int] = None
    ) -> str:
        """Export budget report."""
        summary = self.get_summary(days=days)
        per_lottery = self.get_per_lottery_breakdown(days)
        monthly = self.get_monthly_trend()

        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "summary": summary,
            "per_lottery": per_lottery,
            "monthly_trend": monthly
        }

        if format == "json":
            content = json.dumps(report, indent=2)
        elif format == "csv":
            # Flatten to CSV
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Lottery", "Total Spent", "Total Won", "Net", "ROI %", "Tickets"])
            for p in per_lottery:
                writer.writerow([p["lottery"], p["total_spent"], p["total_won"], p["net"], f"{p['roi']:.2f}", p["total_tickets"]])
            content = output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)

        return content


class MultiDrawPredictor:
    """Generate predictions for multiple upcoming draws with confidence decay modeling."""

    def __init__(self, lottery: str):
        self.lottery = lottery
        from .models import EnsemblePredictor
        self.ensemble = EnsemblePredictor(lottery)
        self.config = LOTTERY_CONFIGS[lottery]

    def predict_multiple_draws(
        self,
        num_draws: int = 5,
        predictions_per_draw: int = 3,
        confidence_decay: float = 0.95
    ) -> List[Dict]:
        """Generate predictions for multiple upcoming draws."""
        all_predictions = []
        base_confidence = 0.1  # Base confidence for ensemble

        for draw_idx in range(num_draws):
            # Get predictions for this draw
            preds = self.ensemble.get_consensus_prediction(num_predictions=predictions_per_draw)

            # Apply confidence decay for future draws
            decay_factor = confidence_decay ** draw_idx
            adjusted_confidence = base_confidence * decay_factor

            draw_predictions = []
            for i, pred in enumerate(preds):
                draw_predictions.append({
                    "set": i + 1,
                    "main_numbers": pred.main_numbers,
                    "bonus_numbers": pred.bonus_numbers,
                    "method": pred.method,
                    "base_confidence": pred.confidence,
                    "adjusted_confidence": min(pred.confidence * decay_factor, 1.0),
                    "draw_index": draw_idx,
                    "decay_factor": decay_factor
                })

            all_predictions.append({
                "draw_number": draw_idx + 1,
                "days_ahead": draw_idx * self._get_draw_interval_days(),
                "predictions": draw_predictions
            })

        return all_predictions

    def _get_draw_interval_days(self) -> int:
        """Estimate days between draws based on draw schedule."""
        draw_days = self.config.get("draw_days", [])
        if len(draw_days) >= 2:
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            days = [day_map.get(d[:3], 0) for d in draw_days]
            days.sort()
            # Find minimum gap
            min_gap = 7
            for i in range(len(days)):
                for j in range(i + 1, len(days)):
                    gap = (days[j] - days[i]) % 7
                    if gap > 0 and gap < min_gap:
                        min_gap = gap
            return min_gap
        return 3  # Default


class VoiceAssistant:
    """Voice assistant integration for lottery predictions."""

    def __init__(self):
        self.intents = {
            "get_predictions": [
                "give me lottery numbers",
                "predict lottery numbers",
                "what are the winning numbers",
                "generate predictions"
            ],
            "get_next_draw": [
                "when is the next draw",
                "next lottery draw",
                "draw schedule"
            ],
            "get_odds": [
                "what are the odds",
                "lottery odds",
                "chances of winning"
            ]
        }

    def parse_intent(self, text: str) -> Optional[str]:
        """Parse user intent from voice/text input."""
        text_lower = text.lower()

        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent
        return None

    def extract_lottery(self, text: str) -> Optional[str]:
        """Extract lottery name from text."""
        text_lower = text.lower()

        lottery_keywords = {
            "powerball": "powerball",
            "mega millions": "mega_millions",
            "mega million": "mega_millions",
            "florida lotto": "florida_lotto",
            "euro millions": "euromillions",
            "euromillions": "euromillions",
            "uk lotto": "uk_lotto",
            "super enalotto": "superenalotto"
        }

        for keyword, key in lottery_keywords.items():
            if keyword in text_lower:
                return key

        return None

    def generate_response(self, intent: str, lottery: Optional[str] = None, predictions: Optional[List] = None) -> str:
        """Generate voice response."""
        if not lottery:
            lottery = "powerball"

        lottery_name = LOTTERY_CONFIGS.get(lottery, {}).get("name", lottery)

        if intent == "get_predictions":
            if predictions:
                pred_text = []
                for i, p in enumerate(predictions[:3], 1):
                    main = " ".join(str(n) for n in p.get("main_numbers", []))
                    bonus = " ".join(str(n) for n in p.get("bonus_numbers", []))
                    pred_text.append(f"Set {i}: {main}. Bonus: {bonus}")
                return f"Here are your {lottery_name} predictions: " + "; ".join(pred_text)
            else:
                return f"I'll generate predictions for {lottery_name}. Please check the app for details."

        elif intent == "get_next_draw":
            return f"The next {lottery_name} draw is scheduled for... Please check the draw calendar for exact time."

        elif intent == "get_odds":
            from .validation import OddsCalculator
            calc = OddsCalculator(lottery)
            odds = calc.full_odds_analysis()
            return f"The jackpot odds for {lottery_name} are {odds.jackpot_odds_str}. The expected value is ${odds.expected_value:.4f}."

        return "I'm not sure what you'd like. You can ask for predictions, the next draw, or odds."

    def process_request(self, text: str) -> Dict:
        """Process a voice/text request and return response."""
        intent = self.parse_intent(text)
        lottery = self.extract_lottery(text)

        if not intent:
            return {
                "success": False,
                "response": "I didn't understand. Try asking for predictions, the next draw, or odds."
            }

        response = self.generate_response(intent, lottery)

        return {
            "success": True,
            "intent": intent,
            "lottery": lottery,
            "response": response
        }


# ============================================================================
# v2.4 - Enhanced History & Analytics
# ============================================================================

@dataclass
class DataHealthRecord:
    """A record of data health metrics for a lottery."""
    timestamp: str
    lottery: str
    total_draws: int
    date_range_start: str
    date_range_end: str
    days_since_last_draw: int
    expected_draws: int
    missing_draws: int
    gaps: List[Dict]
    anomalies: List[Dict]
    completeness_score: float
    freshness_score: float


@dataclass
class CrossLotteryCorrelation:
    """Cross-lottery correlation analysis result."""
    lottery_a: str
    lottery_b: str
    correlation_type: str  # 'number_frequency', 'sum_distribution', 'pair_patterns', 'odd_even'
    correlation_coefficient: float
    p_value: float
    sample_size: int
    significant: bool
    details: Dict


class HistoricalDataHealthDashboard:
    """Monitor data freshness, gaps, anomalies, and backfill scheduling."""

    def __init__(self, storage_path: str = "data_health"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.health_file = self.storage_path / "health_history.json"
        self.records: List[DataHealthRecord] = []
        self._load_records()

    def _load_records(self):
        """Load health records from disk."""
        if self.health_file.exists():
            try:
                with open(self.health_file, 'r') as f:
                    data = json.load(f)
                self.records = [DataHealthRecord(**r) for r in data]
            except Exception:
                pass

    def _save_records(self):
        """Save health records to disk."""
        with open(self.health_file, 'w') as f:
            json.dump([r.__dict__ for r in self.records], f, indent=2)

    def analyze_lottery_health(self, lottery: str) -> DataHealthRecord:
        """Analyze data health for a specific lottery."""
        from .data import get_data_manager, LOTTERY_CONFIGS
        from datetime import datetime, timedelta

        dm = get_data_manager(lottery)
        df = dm.load_data()
        config = LOTTERY_CONFIGS[lottery]

        if len(df) == 0:
            return DataHealthRecord(
                timestamp=datetime.now().isoformat(),
                lottery=lottery,
                total_draws=0,
                date_range_start="",
                date_range_end="",
                days_since_last_draw=999,
                expected_draws=0,
                missing_draws=0,
                gaps=[],
                anomalies=[],
                completeness_score=0.0,
                freshness_score=0.0
            )

        # Basic metrics
        total_draws = len(df)
        date_start = df['date'].min()
        date_end = df['date'].max()

        # Days since last draw
        days_since = (datetime.now() - date_end).days

        # Calculate expected draws based on draw schedule
        draw_days = config.get('draw_days', [])
        expected_draws = self._calculate_expected_draws(date_start, date_end, draw_days)

        # Detect gaps
        gaps = self._detect_gaps(df, draw_days)

        # Detect anomalies using existing validator
        anomalies = self._detect_anomalies(df)

        # Completeness score
        completeness = min(1.0, total_draws / max(1, expected_draws)) if expected_draws > 0 else 1.0

        # Freshness score (1.0 if draw was within expected interval, decays over time)
        max_interval = self._get_max_draw_interval(draw_days)
        freshness = max(0.0, 1.0 - (days_since / (max_interval * 2))) if max_interval > 0 else 0.5

        record = DataHealthRecord(
            timestamp=datetime.now().isoformat(),
            lottery=lottery,
            total_draws=total_draws,
            date_range_start=date_start.strftime('%Y-%m-%d') if hasattr(date_start, 'strftime') else str(date_start),
            date_range_end=date_end.strftime('%Y-%m-%d') if hasattr(date_end, 'strftime') else str(date_end),
            days_since_last_draw=days_since,
            expected_draws=expected_draws,
            missing_draws=max(0, expected_draws - total_draws),
            gaps=gaps,
            anomalies=anomalies,
            completeness_score=completeness,
            freshness_score=freshness
        )

        self.records.append(record)
        # Keep only last 100 records per lottery
        self.records = [r for r in self.records if r.lottery != lottery][-99:] + [record]
        self._save_records()

        return record

    def _calculate_expected_draws(self, start_date, end_date, draw_days) -> int:
        """Calculate expected number of draws based on schedule."""
        if not draw_days:
            return 0

        day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        draw_day_nums = [day_map.get(d[:3], 0) for d in draw_days]

        # Simple estimation: weeks * draws per week
        days_diff = (end_date - start_date).days
        weeks = days_diff / 7
        draws_per_week = len(draw_day_nums)

        return int(weeks * draws_per_week) + 1

    def _get_max_draw_interval(self, draw_days) -> int:
        """Get maximum days between draws."""
        if not draw_days or len(draw_days) < 2:
            return 7

        day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        days = sorted([day_map.get(d[:3], 0) for d in draw_days])

        max_gap = 0
        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                gap = (days[j] - days[i]) % 7
                if gap > 0 and gap < max_gap:
                    max_gap = gap
        return max_gap if max_gap > 0 else 7

    def _detect_gaps(self, df, draw_days) -> List[Dict]:
        """Detect gaps in draw schedule."""
        if len(df) < 2:
            return []

        day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        draw_day_nums = sorted([day_map.get(d[:3], 0) for d in draw_days])

        gaps = []
        dates = df['date'].tolist()

        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]

            if hasattr(prev_date, 'weekday'):
                prev_weekday = prev_date.weekday()
            else:
                prev_weekday = pd.Timestamp(prev_date).weekday()

            if hasattr(curr_date, 'weekday'):
                curr_weekday = curr_date.weekday()
            else:
                curr_weekday = pd.Timestamp(curr_date).weekday()

            days_diff = (curr_date - prev_date).days if hasattr(curr_date, '__sub__') else (pd.Timestamp(curr_date) - pd.Timestamp(prev_date)).days

            # Check if there should have been a draw in between
            expected_next = self._get_next_draw_day(prev_weekday, draw_day_nums)
            if expected_next is not None:
                expected_days = (expected_next - prev_weekday) % 7
                if expected_days == 0:
                    expected_days = 7

                if days_diff > expected_days + 2:  # Allow 2 days tolerance
                    gaps.append({
                        "type": "missing_draw",
                        "expected_date": (prev_date + pd.Timedelta(days=expected_days)).strftime('%Y-%m-%d') if hasattr(prev_date, '__add__') else str(prev_date),
                        "actual_date": curr_date.strftime('%Y-%m-%d') if hasattr(curr_date, 'strftime') else str(curr_date),
                        "gap_days": days_diff,
                        "expected_interval": expected_days
                    })

        return gaps

    def _get_next_draw_day(self, current_weekday, draw_day_nums):
        """Get the next draw day after current weekday."""
        for d in draw_day_nums:
            if d > current_weekday:
                return d
        return draw_day_nums[0] if draw_day_nums else None

    def _detect_anomalies(self, df) -> List[Dict]:
        """Detect anomalies in the data."""
        anomalies = []

        # Check for duplicate draws
        draw_keys = df.apply(lambda row: tuple(sorted(row['main_numbers'])) + tuple(sorted(row['bonus_numbers'])) if row['bonus_numbers'] else tuple(sorted(row['main_numbers'])), axis=1)
        duplicates = draw_keys[draw_keys.duplicated()]
        if len(duplicates) > 0:
            anomalies.append({
                "type": "duplicate_draws",
                "count": len(duplicates),
                "details": f"Found {len(duplicates)} duplicate draw combinations"
            })

        # Check for out-of-range numbers
        from .data import LOTTERY_CONFIGS
        config = LOTTERY_CONFIGS.get(df['lottery'].iloc[0] if 'lottery' in df.columns else 'powerball')
        if config:
            main_max = config['main_max']
            bonus_max = config['bonus_max']

            for idx, row in df.iterrows():
                for n in row['main_numbers']:
                    if n < 1 or n > main_max:
                        anomalies.append({
                            "type": "out_of_range_number",
                            "draw_number": row.get('draw_number', idx),
                            "number": n,
                            "details": f"Main number {n} out of range (1-{main_max})"
                        })
                for n in row['bonus_numbers']:
                    if n < 1 or n > bonus_max:
                        anomalies.append({
                            "type": "out_of_range_number",
                            "draw_number": row.get('draw_number', idx),
                            "number": n,
                            "details": f"Bonus number {n} out of range (1-{bonus_max})"
                        })

        # Check for unusual gaps in dates
        dates = df['date'].tolist()
        if len(dates) > 1:
            intervals = []
            for i in range(1, len(dates)):
                d1 = dates[i-1]
                d2 = dates[i]
                diff = (d2 - d1).days if hasattr(d2, '__sub__') else (pd.Timestamp(d2) - pd.Timestamp(d1)).days
                intervals.append(diff)

            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                for i, diff in enumerate(intervals):
                    if diff > avg_interval + 3 * std_interval:
                        anomalies.append({
                            "type": "unusual_date_gap",
                            "gap_days": diff,
                            "expected_avg": avg_interval,
                            "details": f"Unusually large gap of {diff} days between draws"
                        })

        return anomalies

    def get_health_summary(self, lottery: str) -> Dict:
        """Get health summary for a lottery."""
        records = [r for r in self.records if r.lottery == lottery]
        if not records:
            return {"error": "No health data available"}

        latest = records[-1]
        return {
            "lottery": lottery,
            "total_draws": latest.total_draws,
            "date_range": f"{latest.date_range_start} to {latest.date_range_end}",
            "days_since_last_draw": latest.days_since_last_draw,
            "expected_draws": latest.expected_draws,
            "missing_draws": latest.missing_draws,
            "completeness_score": f"{latest.completeness_score:.1%}",
            "freshness_score": f"{latest.freshness_score:.1%}",
            "gaps_count": len(latest.gaps),
            "anomalies_count": len(latest.anomalies),
            "last_checked": latest.timestamp
        }

    def get_all_lotteries_health(self) -> List[Dict]:
        """Get health summary for all lotteries."""
        lotteries = set(r.lottery for r in self.records)
        results = []
        for lottery in lotteries:
            summary = self.get_health_summary(lottery)
            if "error" not in summary:
                results.append(summary)
        return sorted(results, key=lambda x: x['freshness_score'], reverse=True)

    def schedule_backfill(self, lottery: str, source: str = "auto") -> Dict:
        """Schedule a backfill for missing data."""
        record = self.analyze_lottery_health(lottery)

        if record.missing_draws == 0 and len(record.gaps) == 0:
            return {
                "status": "not_needed",
                "message": "No missing data detected"
            }

        # In a real implementation, this would trigger a background job
        # For now, we return the backfill plan
        return {
            "status": "scheduled",
            "lottery": lottery,
            "missing_draws": record.missing_draws,
            "gaps_to_fill": len(record.gaps),
            "source": source,
            "estimated_completion": "Depends on data source availability"
        }


class CrossLotteryCorrelationAnalyzer:
    """Analyze historical correlations across different lotteries."""

    def __init__(self, storage_path: str = "cross_lottery_correlations"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.correlation_file = self.storage_path / "correlations.json"
        self.results: List[CrossLotteryCorrelation] = []
        self._load_results()

    def _load_results(self):
        """Load correlation results from disk."""
        if self.correlation_file.exists():
            try:
                with open(self.correlation_file, 'r') as f:
                    data = json.load(f)
                self.results = [CrossLotteryCorrelation(**r) for r in data]
            except Exception:
                pass

    def _save_results(self):
        """Save correlation results to disk."""
        with open(self.correlation_file, 'w') as f:
            json.dump([r.__dict__ for r in self.results], f, indent=2)

    def analyze_number_frequency_correlation(self, lottery_a: str, lottery_b: str) -> CrossLotteryCorrelation:
        """Analyze correlation in number frequency patterns between two lotteries."""
        from .data import get_data_manager
        from scipy import stats

        dm_a = get_data_manager(lottery_a)
        dm_b = get_data_manager(lottery_b)

        df_a = dm_a.load_data()
        df_b = dm_b.load_data()

        # Use overlapping date range
        min_date = max(df_a['date'].min(), df_b['date'].min())
        max_date = min(df_a['date'].max(), df_b['date'].max())

        df_a = df_a[(df_a['date'] >= min_date) & (df_a['date'] <= max_date)]
        df_b = df_b[(df_b['date'] >= min_date) & (df_b['date'] <= max_date)]

        # Get number frequencies for common range
        config_a = dm_a.config
        config_b = dm_b.config

        # Find overlapping number range
        max_common = min(config_a['main_max'], config_b['main_max'])

        freq_a = np.zeros(max_common)
        freq_b = np.zeros(max_common)

        for _, row in df_a.iterrows():
            for n in row['main_numbers']:
                if n <= max_common:
                    freq_a[n-1] += 1

        for _, row in df_b.iterrows():
            for n in row['main_numbers']:
                if n <= max_common:
                    freq_b[n-1] += 1

        # Normalize by number of draws
        freq_a = freq_a / max(1, len(df_a))
        freq_b = freq_b / max(1, len(df_b))

        # Calculate correlation
        corr, p_value = stats.pearsonr(freq_a, freq_b)

        # Convert numpy types to Python native types for JSON serialization
        corr = float(corr)
        p_value = float(p_value)
        significant = bool(p_value < 0.05)

        result = CrossLotteryCorrelation(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            correlation_type="number_frequency",
            correlation_coefficient=corr,
            p_value=p_value,
            sample_size=min(len(df_a), len(df_b)),
            significant=significant,
            details={
                "max_common_number": int(max_common),
                "draws_a": int(len(df_a)),
                "draws_b": int(len(df_b)),
                "date_range": f"{min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)} to {max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)}"
            }
        )

        self.results.append(result)
        self._save_results()
        return result

    def analyze_sum_distribution_correlation(self, lottery_a: str, lottery_b: str) -> CrossLotteryCorrelation:
        """Analyze correlation in sum distribution between two lotteries."""
        from .data import get_data_manager
        from scipy import stats

        dm_a = get_data_manager(lottery_a)
        dm_b = get_data_manager(lottery_b)

        df_a = dm_a.load_data()
        df_b = dm_b.load_data()

        # Use overlapping date range
        min_date = max(df_a['date'].min(), df_b['date'].min())
        max_date = min(df_a['date'].max(), df_b['date'].max())

        df_a = df_a[(df_a['date'] >= min_date) & (df_a['date'] <= max_date)]
        df_b = df_b[(df_b['date'] >= min_date) & (df_b['date'] <= max_date)]

        # Calculate sums
        sums_a = df_a['main_numbers'].apply(sum).values
        sums_b = df_b['main_numbers'].apply(sum).values

        # Align by date (use common dates)
        # For simplicity, we'll correlate the distributions
        # Create histograms with same bins
        min_sum = min(sums_a.min(), sums_b.min())
        max_sum = max(sums_a.max(), sums_b.max())
        bins = np.linspace(min_sum, max_sum, 20)

        hist_a, _ = np.histogram(sums_a, bins=bins, density=True)
        hist_b, _ = np.histogram(sums_b, bins=bins, density=True)

        corr, p_value = stats.pearsonr(hist_a, hist_b)

        # Convert numpy types to Python native types for JSON serialization
        corr = float(corr)
        p_value = float(p_value)
        significant = bool(p_value < 0.05)

        result = CrossLotteryCorrelation(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            correlation_type="sum_distribution",
            correlation_coefficient=corr,
            p_value=p_value,
            sample_size=min(len(df_a), len(df_b)),
            significant=significant,
            details={
                "mean_sum_a": float(sums_a.mean()),
                "mean_sum_b": float(sums_b.mean()),
                "std_sum_a": float(sums_a.std()),
                "std_sum_b": float(sums_b.std()),
                "draws_a": int(len(df_a)),
                "draws_b": int(len(df_b))
            }
        )

        self.results.append(result)
        self._save_results()
        return result

    def analyze_odd_even_correlation(self, lottery_a: str, lottery_b: str) -> CrossLotteryCorrelation:
        """Analyze correlation in odd/even patterns between two lotteries."""
        from .data import get_data_manager
        from scipy import stats

        dm_a = get_data_manager(lottery_a)
        dm_b = get_data_manager(lottery_b)

        df_a = dm_a.load_data()
        df_b = dm_b.load_data()

        # Use overlapping date range
        min_date = max(df_a['date'].min(), df_b['date'].min())
        max_date = min(df_a['date'].max(), df_b['date'].max())

        df_a = df_a[(df_a['date'] >= min_date) & (df_a['date'] <= max_date)]
        df_b = df_b[(df_b['date'] >= min_date) & (df_b['date'] <= max_date)]

        # Calculate odd/even ratios per draw
        def get_odd_even_ratio(numbers):
            odds = sum(1 for n in numbers if n % 2 == 1)
            return odds / len(numbers) if numbers else 0

        ratios_a = df_a['main_numbers'].apply(get_odd_even_ratio).values
        ratios_b = df_b['main_numbers'].apply(get_odd_even_ratio).values

        corr, p_value = stats.pearsonr(ratios_a, ratios_b)

        # Convert numpy types to Python native types for JSON serialization
        corr = float(corr)
        p_value = float(p_value)
        significant = bool(p_value < 0.05)

        result = CrossLotteryCorrelation(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            correlation_type="odd_even",
            correlation_coefficient=corr,
            p_value=p_value,
            sample_size=min(len(df_a), len(df_b)),
            significant=significant,
            details={
                "avg_odd_ratio_a": float(ratios_a.mean()),
                "avg_odd_ratio_b": float(ratios_b.mean()),
                "draws_a": int(len(df_a)),
                "draws_b": int(len(df_b))
            }
        )

        self.results.append(result)
        self._save_results()
        return result

    def analyze_all_correlations(self, lottery_a: str, lottery_b: str) -> List[CrossLotteryCorrelation]:
        """Run all correlation analyses between two lotteries."""
        results = []

        try:
            results.append(self.analyze_number_frequency_correlation(lottery_a, lottery_b))
        except Exception as e:
            print(f"Number frequency correlation failed: {e}")

        try:
            results.append(self.analyze_sum_distribution_correlation(lottery_a, lottery_b))
        except Exception as e:
            print(f"Sum distribution correlation failed: {e}")

        try:
            results.append(self.analyze_odd_even_correlation(lottery_a, lottery_b))
        except Exception as e:
            print(f"Odd/even correlation failed: {e}")

        return results

    def get_significant_correlations(self, threshold: float = 0.05) -> List[CrossLotteryCorrelation]:
        """Get all significant correlations."""
        return [r for r in self.results if r.p_value < threshold]

    def find_correlated_lotteries(self, base_lottery: str, top_n: int = 5) -> List[Dict]:
        """Find lotteries most correlated with a base lottery."""
        from .data import LOTTERY_CONFIGS

        other_lotteries = [k for k in LOTTERY_CONFIGS.keys() if k != base_lottery]
        correlations = []

        for other in other_lotteries:
            try:
                results = self.analyze_all_correlations(base_lottery, other)
                # Average correlation across all types
                valid_results = [r for r in results if r.correlation_coefficient is not None]
                if valid_results:
                    avg_corr = np.mean([abs(r.correlation_coefficient) for r in valid_results])
                    min_p = min(r.p_value for r in valid_results)
                    correlations.append({
                        "lottery": other,
                        "name": LOTTERY_CONFIGS[other]['name'],
                        "avg_correlation": avg_corr,
                        "min_p_value": min_p,
                        "significant": min_p < 0.05,
                        "details": {r.correlation_type: {"corr": r.correlation_coefficient, "p": r.p_value} for r in valid_results}
                    })
            except Exception as e:
                print(f"Correlation analysis failed for {base_lottery} vs {other}: {e}")

        correlations.sort(key=lambda x: x['avg_correlation'], reverse=True)
        return correlations[:top_n]