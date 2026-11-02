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