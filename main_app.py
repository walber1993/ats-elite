import streamlit as st
import pandas as pd
import pdfplumber
import spacy
import os
import psycopg2
import re
from psycopg2 import extras
from passlib.hash import pbkdf2_sha256
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF

# ==========================================
# 1. CONFIGURAÇÕES, IA E BANCO DE DADOS INICIAL
# ==========================================

load_dotenv()

# Lógica de Secrets: Prioriza o servidor do Streamlit, mas aceita Local se configurado
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Caso rode localmente sem o arquivo secrets do streamlit, ele tenta pegar do .env
    api_key = os.getenv("OPENAI_API_KEY", "SUA_CHAVE_AQUI_PARA_TESTE_LOCAL")

client = OpenAI(api_key=api_key)

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("pt_core_news_sm")
    except:
        return None

nlp = load_nlp()

# Banco de Dados via Secrets ou Env
if "DATABASE_URL" in st.secrets:
    DB_URL = st.secrets["DATABASE_URL"]
else:
    DB_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg2.connect(
        st.secrets["DB_URL"],
        sslmode="require"
    )

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            phone TEXT,
            plan TEXT DEFAULT 'free',
            credits INTEGER DEFAULT 1,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def gerar_pdf(texto):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    # Substitui caracteres não suportados pelo Latin-1 para evitar erro no FPDF
    texto_limpo = texto.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=texto_limpo)
    return pdf.output(dest='S').encode('latin-1')

def db_cadastrar_usuario(email, senha, nome, telefone):
    conn = get_connection()
    cur = conn.cursor()
    try:
        hash_s = pbkdf2_sha256.hash(senha)
        cur.execute("""
            INSERT INTO users (email, password_hash, full_name, phone) 
            VALUES (%s, %s, %s, %s)
        """, (email, hash_s, nome, telefone))
        conn.commit()
        return True, "Conta criada com sucesso! Faça login para começar."
    except psycopg2.errors.UniqueViolation:
        return False, "Este e-mail já está cadastrado em nossa base."
    except Exception as e:
        return False, f"Erro ao acessar banco de dados: {e}"
    finally:
        cur.close()
        conn.close()

def db_verificar_login(email, senha):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and pbkdf2_sha256.verify(senha, user['password_hash']):
        return dict(user)
    return None

def db_consumir_credito(email):
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE users
        SET credits = credits - 1
        WHERE email = %s AND credits > 0
        RETURNING credits;
    """, (email,))
    
    result = cur.fetchone()
    conn.commit()
    
    cur.close()
    conn.close()
    
    return result

try:
    init_db()
except:
    pass

# ==========================================
# 2. MOTOR ATS: LÓGICA DE MATCH PONDERADO
# ==========================================
CATEGORIAS = {
    "HARD_SKILLS": {"peso": 0.50, "label": "Hard Skills (Técnicas)"},
    "FERRAMENTAS": {"peso": 0.35, "label": "Tecnologias e Softwares"},
    "SOFT_SKILLS": {"peso": 0.15, "label": "Habilidades Comportamentais"}
}

MAPA_FERRAMENTAS = {"crm", "excel", "salesforce", "sap", "python", "sql", "power bi", "slack", "kommo", "pipedrive", "rd station", "hubspot", "trello", "jira", "chatgpt", "metime", "zoom", "whatsapp"}
SOFT_SKILLS_ELITE = {"negociação", "liderança", "comunicação", "oratória", "resolução", "conflitos", "organização", "planejamento", "estratégia", "empatia", "resiliência", "adaptabilidade", "ética", "colaboração"}

def extrair_entidades(texto):
    if not nlp: return {"HARD_SKILLS": set(), "FERRAMENTAS": set(), "SOFT_SKILLS": set()}
    doc = nlp(texto.lower())
    extraidas = {"HARD_SKILLS": set(), "FERRAMENTAS": set(), "SOFT_SKILLS": set()}
    for token in doc:
        word = token.lemma_.strip()
        if word in MAPA_FERRAMENTAS: extraidas["FERRAMENTAS"].add(word)
        elif word in SOFT_SKILLS_ELITE: extraidas["SOFT_SKILLS"].add(word)
        elif len(word) > 3 and not token.is_stop: extraidas["HARD_SKILLS"].add(word)
    return extraidas

def calcular_match(cv_skills, vaga_skills):
    total_score, detalhes = 0, {}
    for cat, info in CATEGORIAS.items():
        v_set, c_set = vaga_skills[cat], cv_skills[cat]
        if not v_set:
            total_score += info["peso"]
            detalhes[cat] = {"match": set(), "falta": set()}
            continue
        matches = c_set & v_set
        total_score += (len(matches) / len(v_set)) * info["peso"]
        detalhes[cat] = {"match": matches, "falta": v_set - c_set}
    return int(total_score * 100), detalhes

# ==========================================
# 3. INTERFACE E UI (STREAMLIT)
# ==========================================
st.set_page_config(page_title="ATS Elite Pro", layout="wide", page_icon="🛡️")

if "user_auth" not in st.session_state: st.session_state.user_auth = None
if "cv_texto" not in st.session_state: st.session_state.cv_texto = None
if "resultado_analise" not in st.session_state: st.session_state.resultado_analise = None
if "cv_otimizado" not in st.session_state: st.session_state.cv_otimizado = None
if "vaga_atual" not in st.session_state: st.session_state.vaga_atual = ""

with st.sidebar:
    st.title("🛡️ ATS Elite v1.0")
    if not st.session_state.user_auth:
        abas_auth = st.tabs(["Login", "Cadastro"])
        with abas_auth[0]:
            e_l = st.text_input("E-mail", key="l_email")
            s_l = st.text_input("Senha", type="password", key="l_pass")
            if st.button("Entrar no Painel", use_container_width=True):
                u = db_verificar_login(e_l, s_l)
                if u:
                    st.session_state.user_auth = u
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
        with abas_auth[1]:
            st.markdown("### Cadastro Gratuito")
            new_nome = st.text_input("Nome Completo", key="c_nome")
            new_email = st.text_input("E-mail Profissional", key="c_email")
            new_phone = st.text_input("WhatsApp (DDD + Número)", key="c_phone")
            new_pass = st.text_input("Crie uma Senha", type="password", key="c_pass")
            if st.button("🚀 Criar Conta e Analisar", use_container_width=True):
                num_clean = re.sub(r'\D', '', new_phone)
                ok, msg = db_cadastrar_usuario(new_email, new_pass, new_nome, num_clean)
                if ok: st.success(msg)
                else: st.error(msg)
    else:
        p_nome = st.session_state.user_auth['full_name'].split()[0]
        credits = st.session_state.user_auth['credits']
        plan = st.session_state.user_auth['plan']
    
        st.write(f"Bem-vindo, **{p_nome}**! 🚀")
        st.metric("Créditos restantes", credits)
    
        # 🔥 USUÁRIO FREE
        if plan == 'free':
    
            if credits > 0:
                st.info(f"Você ainda tem {credits} análise(s) gratuita(s).")
            else:
                st.error("🚫 Seus créditos acabaram.")
    
            st.markdown("### 🚀 Desbloqueie o modo PRO")
            st.markdown("Tenha acesso a **50 otimizações completas de currículo e LinkedIn.**")
    
            if st.button("🔥 Desbloquear por R$29,90"):
                st.markdown(
                    "[👉 Clique aqui para fazer upgrade](SEU_LINK_DE_PAGAMENTO)",
                    unsafe_allow_html=True
                )
    
        # 🔥 USUÁRIO PRO
        if plan != 'free':
            st.success("✅ Você é PRO. Aproveite todos os recursos!")
    
        if st.button("Encerrar Sessão", key="btn_logout"):
            st.session_state.user_auth = None
            st.rerun()

    if st.button("Encerrar Sessão", key="btn_logout"):
        st.session_state.user_auth = None
        st.rerun()

if not st.session_state.user_auth:
    st.header("Sua carreira não pode depender da sorte.")
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?q=80&w=2070", use_container_width=True)
else:
    if st.session_state.cv_texto is None:
        up = st.file_uploader("Upload do Currículo", type="pdf", key="pdf_uploader")
        if up and st.button("Processar Dados do PDF"):
            with pdfplumber.open(up) as pdf:
                st.session_state.cv_texto = "".join([p.extract_text() or "" for p in pdf.pages])
            st.rerun()
    else:
        t1, t2, t3 = st.tabs(["🔍 Match com Vaga", "✍️ Reescrita IA (PRO)", "✨ Estratégia LinkedIn (PRO)"])
        
        with t1:
            if st.session_state.resultado_analise:
                res = st.session_state.resultado_analise
                st.metric("Aderência ATS", f"{res['score']}%")
                c1, c2, c3 = st.columns(3)
                for col, cat in zip([c1, c2, c3], ["FERRAMENTAS", "HARD_SKILLS", "SOFT_SKILLS"]):
                    with col:
                        st.write(f"**{CATEGORIAS[cat]['label']}**")
                        for m in res['detalhes'][cat]["match"]: st.write(f"✅ {m}")
                        for f in res['detalhes'][cat]["falta"]: st.write(f"❌ {f}")
                if st.button("Analisar Outra Vaga"):
                    st.session_state.resultado_analise = None
                    st.rerun()
            else:
                txt_vaga = st.text_area("Cole a descrição da vaga alvo:", height=250)
                if st.button("🔍 Calcular Match Ponderado"):
                    if txt_vaga:
                        # 🔒 CONSUME CRÉDITO COM VALIDAÇÃO
                        result = db_consumir_credito(st.session_state.user_auth['email'])
                
                        if not result:
                            st.error("🚫 Você não possui créditos disponíveis.")
                            st.stop()
                
                        # Atualiza crédito na sessão
                        st.session_state.user_auth['credits'] = result[0]
                
                        st.session_state.vaga_atual = txt_vaga 
                        
                        cv_ent = extrair_entidades(st.session_state.cv_texto)
                        v_ent = extrair_entidades(txt_vaga)
                        score, det = calcular_match(cv_ent, v_ent)
                
                        st.session_state.resultado_analise = {
                            "score": score,
                            "detalhes": det
                        }
                
                        st.rerun()

        with t2:
            if st.session_state.user_auth['plan'] == 'free':
                st.warning("A Reescrita Inteligente é exclusiva para membros **PRO**.")
            else:
                if st.button("🚀 Gerar Currículo Otimizado", key="btn_reescrita_pro"):
                    v_ent = extrair_entidades(st.session_state.vaga_atual)
                    skills_foco = ", ".join(v_ent["HARD_SKILLS"] | v_ent["FERRAMENTAS"])
                    
                    with st.spinner("IA reescrevendo com foco em palavras-chave..."):
                        prompt = f"""
                        Atue como Especialista em Recrutamento. Reescreva o currículo para a vaga fornecida.
                        ⚠️ OBRIGATÓRIO incluir estes termos para o algoritmo aumentar o score: {skills_foco}
                        
                        CV: {st.session_state.cv_texto}
                        VAGA: {st.session_state.vaga_atual}
                        """
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.session_state.cv_otimizado = response.choices[0].message.content
                
                if st.session_state.cv_otimizado:
                    st.markdown(st.session_state.cv_otimizado)
                    pdf_bytes = gerar_pdf(st.session_state.cv_otimizado)
                    st.download_button(
                        label="📥 Baixar Currículo em PDF",
                        data=pdf_bytes,
                        file_name="Curriculo_Otimizado_Elite.pdf",
                        mime="application/pdf"
                    )

        with t3:
            if st.session_state.user_auth['plan'] == 'free':
                st.warning("Funcionalidade exclusiva para membros **PRO**.")
            else:
                if st.button("Gerar Títulos e Resumo SEO", key="btn_ai_pro"):
                    with st.spinner("Criando estratégia..."):
                        p = f"Gere títulos de LinkedIn e um resumo focado em SEO para o CV: {st.session_state.cv_texto}"
                        r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": p}])
                        st.markdown(r.choices[0].message.content)

        st.divider()
        if st.button("🔄 Substituir Currículo Atual", key="btn_replace_cv"):
            st.session_state.cv_texto = None
            st.session_state.resultado_analise = None
            st.session_state.cv_otimizado = None
            st.rerun()
