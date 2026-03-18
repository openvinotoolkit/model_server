#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

from openai import OpenAI

CHINESE_PROMPTS = [
    "Kokoro 是一系列体积虽小但功能强大的 TTS 模型。",
    "今天天气很好，我们去公园散步吧。",
    "请把这份报告在下午三点前发送给我。",
    "人工智能正在改变我们的工作和生活方式。",
    "这个模型支持多种语言，包括中文、英文和日语。",
    "如果出现错误，请重试或联系管理员。",
    "会议将在下周二上午九点准时开始。",
    "请确认你的收货地址和联系电话是否正确。",
    "项目进度需要每周更新一次并提交给经理。",
    "为了提高性能，我们需要优化推理流程。",
    "系统已成功部署到生产环境，监控正常。",
    "这句话用于测试中文语音合成的清晰度。",
    "请阅读并同意服务条款和隐私政策。",
    "模型输出的音频应当自然流畅且可理解。",
    "数据备份已完成，请检查日志确认结果。",
    "请在两分钟内完成系统重启，并确认服务恢复。",
    "客户反馈延迟较高，我们需要检查网络链路。",
    "今天是星期五，记得提交本周的工作总结。",
    "该功能已进入灰度发布阶段，请关注指标变化。",
    "日志中出现多次超时错误，请检查依赖服务。",
    "请将版本号更新为 1.2.3，并生成发布说明。",
    "系统负载过高，建议临时扩容两台实例。",
    "请确认验证码已发送到用户手机号。",
    "数据库备份完成后请验证备份完整性。",
    "优化缓存命中率可以显著提升响应速度。",
    "模型推理耗时过长，需要排查瓶颈。",
    "我们计划在下月上线新的支付流程。",
    "请核对发票信息，确保金额与订单一致。",
    "设备离线超过 30 分钟，请检查供电。",
    "用户输入包含特殊字符，请做好过滤处理。",
    "请确认接口文档已同步更新。",
    "今天的会议取消，改为周三上午十点。",
    "该功能支持多语言切换，请验证中文显示。",
    "请检查邮件是否被误判为垃圾邮件。",
    "请在测试环境验证修复结果，再合入主干。",
]
ENGLISH_PROMPTS = [
    "The HTTP 404 error indicates the requested resource wasn't found on the server.",
    "My phone number is +1-555-123-4567, and my email is john.doe@example.com.",
    "The CPU utilization reached 98.7% at 3:42 AM, triggering an automated alert.",
    "Professor Smith's lecture on quantum mechanics is scheduled for December 15th, 2024.",
    "The API endpoint responds in approximately 127 milliseconds with a 200 OK status.",
    "NVIDIA's GeForce RTX 4090 GPU features 24GB of GDDR6X memory.",
    "The PostgreSQL database crashed at 10:15 PM UTC due to out-of-memory errors.",
    "Dr. Williams recommended acetaminophen 500mg three times daily for pain management.",
    "The SHA-256 hash of the file is 3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9q0r1s2t3u4v.",
    "Mount Everest's peak stands at 8,848.86 meters or 29,031.7 feet above sea level.",
    "The XML configuration file references namespace xmlns:xsi='http://www.w3.org'.",
    "Flight BA2490 departed London Heathrow at 14:25 GMT, arriving in New York JFK at 17:15 EST.",
    "The Schrödinger equation describes quantum-mechanical wave functions: iℏ ∂Ψ/∂t = ĤΨ.",
    "NASA's Artemis III mission aims to land astronauts near the lunar south pole.",
    "The DNS server at IP address 192.168.1.1 failed to resolve www.example.com.",
    "JavaScript's async/await syntax simplifies promise-based asynchronous code handling.",
    "The Wi-Fi password is 'MyS3cur3P@ssw0rd!' with uppercase, numbers, and special characters.",
    "Mrs. O'Brien's restaurant serves crème brûlée and jalapeño poppers as appetizers.",
    "The cryptocurrency wallet address is 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb5.",
    "Tokyo's coordinates are 35.6762° N latitude and 139.6503° E longitude.",
    "The RESTful API uses OAuth 2.0 authentication with JWT bearer tokens.",
    "Linux kernel version 6.5.13 includes patches for CVE-2024-12345 vulnerability.",
    "The chemical formula for sulfuric acid is H₂SO₄, commonly used in batteries.",
    "Amazon Web Services S3 bucket 'prod-data-backup-2024' exceeded 15TB storage.",
    "The SQL query 'SELECT * FROM users WHERE id IN (1,2,3)' returned zero rows.",
    "Dr. García's Ph.D. thesis explored non-Euclidean geometry in n-dimensional spaces.",
    "The HTTPS certificate for *.mydomain.com expires on March 31st at 11:59 PM.",
    "UNESCO's World Heritage site #1347 was designated in Kyoto, Japan.",
    "The docker-compose.yml file defines three microservices: frontend, backend, and Redis cache.",
    "Mr. Müller's BMW X5 accelerates from 0 to 60 mph in just 4.3 seconds.",
    "The grep command 'grep -r TODO *.py | wc -l' found 237 occurrences.",
    "Wellington, New Zealand's capital, experiences winds exceeding 50 km/h regularly.",
    "The TCP/IP packet loss rate increased to 12.4% during the 10 PM network spike.",
    "Einstein's E=mc² equation relates energy, mass, and the speed of light squared.",
    "The JSON payload contains nested arrays: {'users': [{'id': 1, 'name': 'Alice'}]}.",
]

ITALIAN_PROMPTS = [
    "Kokoro è una serie di modelli TTS leggeri ma potenti.",
    "Oggi il tempo è bello, andiamo a fare una passeggiata nel parco.",
    "Per favore, invia questo rapporto entro le tre del pomeriggio.",
    "L'intelligenza artificiale sta cambiando il modo in cui lavoriamo e viviamo.",
    "Questo modello supporta più lingue, inclusi inglese, cinese e giapponese.",
    "Se si verifica un errore, prova di nuovo o contatta l'amministratore.",
    "La riunione inizierà alle nove in punto martedì prossimo.",
    "Per favore, conferma che l'indirizzo di spedizione e il numero di telefono siano corretti.",
    "Lo stato del progetto deve essere aggiornato settimanalmente e inviato al responsabile.",
    "Per migliorare le prestazioni, è necessario ottimizzare la pipeline di inferenza.",
    "Il sistema è stato distribuito con successo nell'ambiente di produzione.",
    "Questa frase viene utilizzata per testare la chiarezza della sintesi vocale italiana.",
    "Si prega di leggere e accettare i termini di servizio e l'informativa sulla privacy.",
    "L'audio di output del modello dovrebbe essere naturale e facile da comprendere.",
    "Il backup dei dati è stato completato, verificare i log per confermare i risultati.",
    "Si prega di completare il riavvio del sistema entro due minuti.",
    "Il feedback dei clienti mostra una latenza elevata, è necessario controllare il collegamento di rete.",
    "Oggi è venerdì, ricorda di inviare il riepilogo del lavoro di questa settimana.",
    "Questa funzione è entrata nella fase di rilascio graduato.",
    "Nel log compaiono più errori di timeout, si prega di controllare i servizi dipendenti.",
    "Si prega di aggiornare il numero di versione a 1.2.3 e generare le note di rilascio.",
    "Il carico del sistema è troppo elevato, si consiglia di espandere temporaneamente due istanze.",
    "Si prega di confermare che il codice di verifica sia stato inviato al telefono dell'utente.",
    "Dopo il backup del database, verificare l'integrità del backup.",
    "L'ottimizzazione del tasso di hit della cache può migliorare significativamente la velocità di risposta.",
    "L'inferenza del modello impiega troppo tempo, è necessario indagare il collo di bottiglia.",
    "Stiamo pianificando il lancio di un nuovo processo di pagamento il mese prossimo.",
    "Si prega di verificare le informazioni della fattura per assicurarsi che l'importo corrisponda all'ordine.",
    "Il dispositivo è offline da più di 30 minuti, si prega di controllare l'alimentazione.",
    "L'input dell'utente contiene caratteri speciali, si prega di gestire il filtraggio correttamente.",
    "Si prega di confermare che la documentazione dell'interfaccia sia stata sincronizzata.",
    "La riunione di oggi è annullata, riprogrammata per mercoledì alle 10.",
    "Questa funzione supporta il cambio di più lingue, si prega di verificare la visualizzazione italiana.",
    "Si prega di controllare se l'email è stata identificata erroneamente come spam.",
    "Si prega di verificare la correzione nell'ambiente di test prima di unire al ramo principale.",
]

SPANISH_PROMPTS = [
    "Kokoro es una serie de modelos TTS ligeros pero potentes.",
    "Hoy el clima es agradable, vamos a pasear por el parque.",
    "Por favor, envía este informe antes de las tres de la tarde.",
    "La inteligencia artificial está cambiando la forma en que trabajamos y vivimos.",
    "Este modelo admite varios idiomas, incluidos inglés, chino y japonés.",
    "Si ocurre un error, intenta de nuevo o comunícate con el administrador.",
    "La reunión comenzará puntualmente a las nueve el martes próximo.",
    "Por favor, confirma que tu dirección de envío y número de teléfono sean correctos.",
    "El progreso del proyecto debe actualizarse semanalmente y enviarse al gerente.",
    "Para mejorar el rendimiento, necesitamos optimizar la tubería de inferencia.",
    "El sistema se ha implementado correctamente en el entorno de producción.",
    "Esta oración se utiliza para probar la claridad de la síntesis de voz en español.",
    "Por favor, lee y acepta los términos de servicio y la política de privacidad.",
    "La salida de audio del modelo debe ser natural y fácil de entender.",
    "La copia de seguridad de datos se ha completado, verifica los registros para confirmar los resultados.",
    "Por favor, completa el reinicio del sistema en dos minutos.",
    "Los comentarios de los clientes muestran alta latencia, necesitamos verificar el enlace de red.",
    "Hoy es viernes, recuerda enviar el resumen del trabajo de esta semana.",
    "Esta función ha entrado en fase de lanzamiento gradual.",
    "Varios errores de tiempo de espera aparecen en los registros, verifica los servicios dependientes.",
    "Por favor, actualiza el número de versión a 1.2.3 y genera las notas de la versión.",
    "La carga del sistema es demasiado alta, se sugiere expandir temporalmente dos instancias.",
    "Por favor, confirma que el código de verificación se haya enviado al teléfono del usuario.",
    "Después de completar la copia de seguridad de la base de datos, verifica la integridad.",
    "Optimizar la tasa de aciertos de caché puede mejorar significativamente la velocidad de respuesta.",
    "La inferencia del modelo toma demasiado tiempo, necesitamos investigar el cuello de botella.",
    "Planeamos lanzar un nuevo proceso de pago el próximo mes.",
    "Por favor, verifica la información de la factura para asegurar que el monto coincida con el pedido.",
    "El dispositivo ha estado desconectado durante más de 30 minutos, verifica el suministro de energía.",
    "La entrada del usuario contiene caracteres especiales, maneja el filtrado correctamente.",
    "Por favor, confirma que la documentación de la interfaz se haya sincronizado.",
    "La reunión de hoy se ha cancelado, reprogramada para el miércoles a las 10 de la mañana.",
    "Esta función admite el cambio de varios idiomas, verifica la pantalla en español.",
    "Por favor, verifica si el correo ha sido identificado erróneamente como spam.",
    "Por favor, verifica la corrección en el entorno de prueba antes de fusionar con la rama principal.",
]

GERMAN_PROMPTS = [
    "Kokoro ist eine Reihe leichter, aber leistungsstarker TTS-Modelle.",
    "Heute ist das Wetter schön, lass uns im Park spazieren gehen.",
    "Bitte sende diesen Bericht bis drei Uhr nachmittags.",
    "Künstliche Intelligenz verändert unsere Arbeits- und Lebensweise.",
    "Dieses Modell unterstützt mehrere Sprachen, darunter Englisch, Chinesisch und Japanisch.",
    "Wenn ein Fehler auftritt, versuche es erneut oder kontaktiere den Administrator.",
    "Die Besprechung beginnt nächsten Dienstag pünktlich um neun Uhr.",
    "Bitte bestätige, dass deine Lieferadresse und Telefonnummer korrekt sind.",
    "Der Projektfortschritt muss wöchentlich aktualisiert und an den Manager gesendet werden.",
    "Um die Leistung zu verbessern, müssen wir die Inferenz-Pipeline optimieren.",
    "Das System wurde erfolgreich in der Produktionsumgebung bereitgestellt.",
    "Dieser Satz wird verwendet, um die Klarheit der deutschen Sprachsynthese zu testen.",
    "Bitte lies und akzeptiere die Nutzungsbedingungen und die Datenschutzrichtlinie.",
    "Die Audioausgabe des Modells sollte natürlich und leicht verständlich sein.",
    "Die Datensicherung wurde abgeschlossen, bitte prüfe die Protokolle zur Bestätigung.",
    "Bitte führe den Neustart des Systems innerhalb von zwei Minuten durch.",
    "Das Kundenfeedback zeigt eine hohe Latenz, wir müssen die Netzwerkverbindung überprüfen.",
    "Heute ist Freitag, denk daran, die Zusammenfassung dieser Woche einzureichen.",
    "Diese Funktion befindet sich jetzt in der schrittweisen Freigabephase.",
    "In den Protokollen erscheinen mehrere Zeitüberschreitungsfehler, bitte prüfe abhängige Dienste.",
    "Bitte aktualisiere die Versionsnummer auf 1.2.3 und erstelle die Versionshinweise.",
    "Die Systemlast ist zu hoch, es wird empfohlen, vorübergehend zwei Instanzen zu erweitern.",
    "Bitte bestätige, dass der Bestätigungscode an die Telefonnummer des Benutzers gesendet wurde.",
    "Nach Abschluss der Datenbanksicherung bitte die Integrität des Backups prüfen.",
    "Die Optimierung der Cache-Trefferquote kann die Reaktionsgeschwindigkeit deutlich verbessern.",
    "Die Modellinferenz dauert zu lange, wir müssen den Engpass untersuchen.",
    "Wir planen, im nächsten Monat einen neuen Zahlungsprozess einzuführen.",
    "Bitte überprüfe die Rechnungsinformationen, damit der Betrag mit der Bestellung übereinstimmt.",
    "Das Gerät ist seit mehr als 30 Minuten offline, bitte prüfe die Stromversorgung.",
    "Die Benutzereingabe enthält Sonderzeichen, bitte behandle die Filterung korrekt.",
    "Bitte bestätige, dass die Schnittstellendokumentation synchronisiert wurde.",
    "Das heutige Meeting wurde abgesagt und auf Mittwoch um zehn Uhr verschoben.",
    "Diese Funktion unterstützt den Wechsel zwischen mehreren Sprachen, bitte prüfe die deutsche Anzeige.",
    "Bitte prüfe, ob die E-Mail fälschlicherweise als Spam markiert wurde.",
    "Bitte verifiziere die Korrektur in der Testumgebung, bevor du in den Hauptzweig zusammenführst.",
]

CHINESE_PUNCT = "，。！？；：、""''（）《》【】—…·、"
LATIN_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def get_prompts_for_language(language: str) -> list[str]:
    """Get prompt list based on language code."""
    language = language.lower()
    if language in ("zh", "zh-cn"):
        return CHINESE_PROMPTS
    elif language in ("en", "en-us", "en-gb"):
        return ENGLISH_PROMPTS
    elif language in ("it", "it-it"):
        return ITALIAN_PROMPTS
    elif language in ("es", "es-es"):
        return SPANISH_PROMPTS
    elif language in ("de", "de-de"):
        return GERMAN_PROMPTS
    else:
        # Default to English for unknown languages
        return ENGLISH_PROMPTS


def normalize_text(text: str, language: str = "en") -> str:
    if not text:
        return ""
    text = text.strip().lower()
    
    language = language.lower()
    if language in ("zh", "zh-cn"):
        # Chinese: remove Chinese punctuation and whitespace
        remove_chars = set(CHINESE_PUNCT)
        remove_chars.update({" ", "\t", "\n", "\r"})
        for ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
            remove_chars.add(ch)
    else:
        # English, Italian, Spanish, German: remove Latin punctuation and whitespace
        remove_chars = set(LATIN_PUNCT)
        remove_chars.update({" ", "\t", "\n", "\r"})
        # Also remove any Chinese punctuation that might appear
        remove_chars.update(set(CHINESE_PUNCT))
    
    return "".join(ch for ch in text if ch not in remove_chars)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def cer(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    prev = list(range(hyp_len + 1))
    curr = [0] * (hyp_len + 1)

    for i in range(1, ref_len + 1):
        curr[0] = i
        r_char = reference[i - 1]
        for j in range(1, hyp_len + 1):
            h_char = hypothesis[j - 1]
            cost = 0 if r_char == h_char else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    distance = prev[hyp_len]
    return distance / ref_len


def tts_request(endpoint: str, model: str, voice: str, prompt: str, language: str) -> bytes:
    url = endpoint.rstrip("/") + "/audio/speech"
    payload = {
        "model": model,
        "voice": voice,
        "input": prompt,
    }
    if language:
        payload["language"] = language
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？；\n":
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []
    if buf:
        sentence = "".join(buf).strip()
        if sentence:
            sentences.append(sentence)

    chunks = []
    current = ""
    for s in sentences:
        if not current:
            current = s
            continue
        if len(current) + len(s) <= max_chars:
            current += s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    if not chunks:
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send prompts to TTS, then transcribe and compare results."
    )
    parser.add_argument("--endpoint", required=True, help="Base URL, e.g. http://localhost:8122/v3")
    parser.add_argument("--tts-model", default="kokoro", help="TTS model name")
    parser.add_argument("--asr-model", default="whisper", help="ASR model name")
    parser.add_argument("--voice", default=None, help="Voice name")
    parser.add_argument("--language", default="en", help="Language code (default: en, options: en, zh, it, es, de)")
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output-dir", default="tts_asr_output", help="Output directory")
    parser.add_argument("--save-audio", action="store_true", help="Save WAV files to output directory")
    parser.add_argument("--text", default=None, help="Single text to send (overrides prompt list)")
    parser.add_argument("--text-file", default=None, help="Path to a text file to send (overrides prompt list)")
    parser.add_argument("--max-chars", type=int, default=300, help="Max chars per TTS request for single text")
    args = parser.parse_args()

    if args.limit <= 0:
        print("--limit must be > 0", file=sys.stderr)
        return 2

    if args.text and args.text_file:
        print("Use only one of --text or --text-file", file=sys.stderr)
        return 2

    if args.text_file:
        try:
            with open(args.text_file, "r", encoding="utf-8") as f:
                single_text = f.read().strip()
        except OSError as exc:
            print(f"Failed to read --text-file: {exc}", file=sys.stderr)
            return 2
        if not single_text:
            print("--text-file is empty", file=sys.stderr)
            return 2
        prompts = split_text_into_chunks(single_text, args.max_chars)
    elif args.text:
        prompts = split_text_into_chunks(args.text, args.max_chars)
    else:
        # Get prompts based on language
        prompts = get_prompts_for_language(args.language)
        if args.limit < len(prompts):
            random.seed(args.seed)
            prompts = random.sample(prompts, args.limit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.endpoint, api_key="unused")

    results = []
    total_tts_time = 0.0
    total_asr_time = 0.0
    
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] {prompt}")

        wav_path = out_dir / f"{idx:02d}.wav"
        try:
            tts_start = time.time()
            audio_bytes = tts_request(
                endpoint=args.endpoint,
                model=args.tts_model,
                voice=args.voice,
                prompt=prompt,
                language=args.language,
            )
            tts_time = time.time() - tts_start
            total_tts_time += tts_time
            
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as exc:
            print(f"  TTS failed: {exc}", file=sys.stderr)
            results.append({
                "prompt": prompt,
                "transcript": "",
                "similarity": 0.0,
                "tts_time": 0.0,
                "asr_time": 0.0,
                "error": f"tts: {exc}",
            })
            continue

        try:
            asr_start = time.time()
            with open(wav_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=args.asr_model,
                    file=audio_file,
                )
            asr_time = time.time() - asr_start
            total_asr_time += asr_time
            transcript_text = transcript.text or ""
        except Exception as exc:
            print(f"  ASR failed: {exc}", file=sys.stderr)
            results.append({
                "prompt": prompt,
                "transcript": "",
                "similarity": 0.0,
                "tts_time": tts_time,
                "asr_time": 0.0,
                "error": f"asr: {exc}",
            })
            if not args.save_audio and wav_path.exists():
                wav_path.unlink(missing_ok=True)
            continue

        n_prompt = normalize_text(prompt, args.language)
        n_trans = normalize_text(transcript_text, args.language)
        score = similarity(n_prompt, n_trans)
        cer_score = cer(n_prompt, n_trans)

        results.append({
            "prompt": prompt,
            "transcript": transcript_text,
            "similarity": score,
            "cer": cer_score,
            "tts_time": tts_time,
            "asr_time": asr_time,
            "error": "",
        })

        print(f"  Transcript: {transcript_text}")
        print(f"  Similarity: {score:.3f}")
        print(f"  CER: {cer_score:.3f}")
        print(f"  TTS time: {tts_time:.3f}s, ASR time: {asr_time:.3f}s\n")

        if not args.save_audio and wav_path.exists():
            wav_path.unlink(missing_ok=True)

    if results:
        avg = sum(r["similarity"] for r in results) / len(results)
        avg_cer = sum(r.get("cer", 1.0) for r in results) / len(results)
    else:
        avg = 0.0
        avg_cer = 1.0

    exact = sum(1 for r in results if normalize_text(r["prompt"], args.language) == normalize_text(r["transcript"], args.language))
    
    avg_tts_time = total_tts_time / len(results) if results else 0.0
    avg_asr_time = total_asr_time / len(results) if results else 0.0
    
    print("=" * 60)
    print(f"Completed {len(results)} items")
    print(f"Exact matches: {exact}")
    print(f"Average similarity: {avg:.3f}")
    print(f"Average CER: {avg_cer:.3f}")
    print(f"Average TTS time: {avg_tts_time:.3f}s")
    print(f"Average ASR time: {avg_asr_time:.3f}s")
    print(f"Total TTS time: {total_tts_time:.3f}s")
    print(f"Total ASR time: {total_asr_time:.3f}s")
    print(f"Total processing time: {total_tts_time + total_asr_time:.3f}s")
    print("Output directory:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
