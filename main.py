
import streamlit as st
import base64
import os
import requests
import json
import time
import re
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import requests
import xml.etree.ElementTree as ET

class CyberSecurityAssistant:
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=groq_api_key,
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are CyberGeni, an ethical hacking trainer who helps students understand cybersecurity concepts and practices. 
                You assist students by generating explanations, code snippets, suggesting tools, or providing step-by-step instructions based on the task and scenario provided.
                
                All information you provide should be for educational purposes only. Emphasize ethical practices and legal boundaries in all responses.
                Always remind users that any security testing should only be performed on systems they own or have explicit permission to test.
                
                Your expertise includes:
                - Network security and penetration testing
                - Web application security
                - Security tools and frameworks
                - Vulnerability assessment techniques
                - Defensive security practices and countermeasures
                - Common security tools commands and usage
                - Google dorking techniques
                - CVE understanding and analysis
                - Latest cybersecurity trends and news
                
                When providing technical guidance:
                1. Explain the concepts first
                2. Outline the methodology
                3. Share relevant code or commands with explanations
                4. Highlight security considerations and ethical implications
                """
            ),
            ("human", "{query}")
        ])
        
        # Create the processing chain
        self.chain = self.prompt | self.llm
        
        # Store conversation history
        self.conversation_history = []
        
    def query(self, user_query):
        """Process a user query and return a response"""
        try:
            # Add query to conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            
            # Process the query
            response = self.chain.invoke({"query": user_query})
            response_text = response.content
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            return f"Error processing your query: {str(e)}"
            
    def search_cve(self, cve_query, limit=5):
        """Search for CVE information"""
        try:
            # Format to search for specific CVE ID or general search
            if re.match(r'CVE-\d{4}-\d{4,}', cve_query, re.IGNORECASE):
                cve_id = cve_query.upper()
                url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
            else:
                # General keyword search
                url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={cve_query}&resultsPerPage={limit}"
            
            headers = {
                "User-Agent": "CyberGeni Educational Tool/1.0"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process the response to extract relevant information
                results = []
                if 'vulnerabilities' in data:
                    for vuln in data['vulnerabilities'][:limit]:
                        cve_item = vuln['cve']
                        cve_id = cve_item['id']
                        
                        # Get description
                        descriptions = cve_item.get('descriptions', [])
                        description = next((desc['value'] for desc in descriptions if desc['lang'] == 'en'), "No description available")
                        
                        # Get severity if available
                        metrics = cve_item.get('metrics', {})
                        cvss_data = metrics.get('cvssMetricV31', [{}])[0] if 'cvssMetricV31' in metrics else metrics.get('cvssMetricV30', [{}])[0] if 'cvssMetricV30' in metrics else {}
                        base_score = cvss_data.get('cvssData', {}).get('baseScore', "N/A") if cvss_data else "N/A"
                        severity = cvss_data.get('cvssData', {}).get('baseSeverity', "N/A") if cvss_data else "N/A"
                        
                        # Get published and last modified dates
                        published = cve_item.get('published', "N/A")
                        last_modified = cve_item.get('lastModified', "N/A")
                        
                        # Format dates if they exist
                        if published != "N/A":
                            published = published.split('T')[0]  # Just take the date part
                        if last_modified != "N/A":
                            last_modified = last_modified.split('T')[0]  # Just take the date part
                        
                        results.append({
                            'cve_id': cve_id,
                            'description': description,
                            'severity': severity,
                            'base_score': base_score,
                            'published': published,
                            'last_modified': last_modified,
                            'references': cve_item.get('references', [])
                        })
                
                if not results:
                    return f"No CVE entries found for '{cve_query}'"
                
                return results
            else:
                return f"Error searching CVE database: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error searching CVE database: {str(e)}"
    
    def get_hacker_news(self, limit=5):
        """Fetch and summarize latest cybersecurity news from The Hacker News using requests and xml.etree"""
        try:
            feed_url = "https://feeds.feedburner.com/TheHackersNews"
            response = requests.get(feed_url)
            
            if response.status_code != 200:
                return "Unable to fetch cybersecurity news at this time."
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'rss': 'http://www.w3.org/2005/Atom'}  # default Atom ns

            news_items = []
            entries = root.findall(".//item")
            
            for entry in entries[:limit]:
                title = entry.findtext("title")
                link = entry.findtext("link")
                published = entry.findtext("pubDate")
                summary = entry.findtext("description")

                # Clean summary from HTML tags
                summary = re.sub(r'<.*?>', '', summary)

                # Date formatting
                try:
                    pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
                    published_formatted = pub_date.strftime("%Y-%m-%d")
                except:
                    published_formatted = published

                # Generate LLM summary
                summary_prompt = f"Summarize this cybersecurity news article in 2-3 sentences, highlighting key security implications:\n\nTitle: {title}\n\nContent: {summary}"

                # Process with LLM
                summary_response = self.llm.invoke(summary_prompt)
                llm_summary = summary_response.content

                news_items.append({
                    'title': title,
                    'published': published_formatted,
                    'link': link,
                    'summary': llm_summary
                })

        
            return news_items

        except Exception as e:
            return f"Error fetching cybersecurity news: {str(e)}"
    
    def generate_tool_commands(self, tool_name, task_description):
        """Generate commands for common security tools based on the task"""
        try:
            prompt = f"""Generate practical command examples for using the security tool '{tool_name}' to accomplish the following task: {task_description}.
            
            Format your response as follows:
            1. Brief introduction to what {tool_name} is and how it's useful for this task
            2. 3-5 practical command examples with syntax highlighting
            3. Explanation of key parameters and options
            4. Common usage scenarios
            5. Security considerations and ethical usage reminder
            
            Be detailed and provide real commands that would actually work in a security testing environment.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating tool commands: {str(e)}"
    
    def generate_google_dorks(self, target_type, objective):
        """Generate Google dorking queries for specific objectives"""
        try:
            prompt = f"""Generate 5-7 effective Google dorking queries that could be used to find {target_type} with the objective of {objective}.
            
            For each query:
            1. Show the exact Google dork syntax
            2. Explain what the query does and why it's effective
            3. Note any specific operators used
            
            Include a brief introduction explaining Google dorking, its legitimate security applications, and ethical considerations.
            
            Reminder: These are for educational purposes and security assessment only.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating Google dorks: {str(e)}"


    def analyze_network_traffic(self, pcap_data=None, sample_type="http"):
        """Analyze network traffic patterns or provided PCAP samples
        
        Args:
            pcap_data: Optional base64 encoded PCAP data or description of traffic
            sample_type: Type of traffic to analyze (http, dns, tcp, etc.)
        """
        try:
            prompt = f"""Analyze the following {sample_type} network traffic scenario and provide security insights:
            
            {pcap_data if pcap_data else f'Provide a detailed analysis of common {sample_type} traffic patterns and security implications'}
            
            In your analysis include:
            1. Key indicators to look for in this type of traffic
            2. Common security issues or attack patterns in {sample_type} traffic
            3. How to identify anomalies or malicious activities
            4. Tools and techniques for monitoring this traffic type
            5. Recommended security controls and best practices
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing network traffic: {str(e)}"
    
    def generate_security_policy(self, organization_type, focus_area):
        """Generate security policy templates based on organization type and focus area
        
        Args:
            organization_type: Type of organization (healthcare, finance, education, etc.)
            focus_area: Specific security focus (network, data, access control, etc.)
        """
        try:
            prompt = f"""Generate a comprehensive security policy template for a {organization_type} organization, 
            focusing specifically on {focus_area} security.
            
            The policy should include:
            1. Purpose and scope
            2. Roles and responsibilities
            3. Specific policy statements and controls
            4. Compliance requirements relevant to {organization_type} organizations
            5. Implementation guidance
            6. Monitoring and enforcement mechanisms
            7. Review and update procedures
            
            Format the policy in a professional, structured manner suitable for organizational use.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating security policy: {str(e)}"
    
    def analyze_attack_scenario(self, attack_vector, target_system):
        """Analyze a security attack scenario and provide defense strategies
        
        Args:
            attack_vector: Type of attack (phishing, ransomware, SQLi, etc.)
            target_system: Type of system being targeted (web server, API, etc.)
        """
        try:
            prompt = f"""Analyze the following attack scenario and provide a comprehensive defense strategy:
            
            Attack Vector: {attack_vector}
            Target System: {target_system}
            
            Your analysis should include:
            1. Attack methodology and typical progression
            2. Initial compromise techniques
            3. Potential lateral movement and persistence mechanisms
            4. Key vulnerabilities exploited
            5. Detection strategies and indicators of compromise
            6. Prevention measures (before attack)
            7. Mitigation strategies (during attack)
            8. Recovery procedures (after attack)
            9. Relevant security tools and frameworks for defense
            
            Keep your recommendations ethical and focused on defensive security.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing attack scenario: {str(e)}"
    
    def decode_or_encode_data(self, input_data, operation, format_type):
        """Encode or decode data in various formats
        
        Args:
            input_data: The data to encode/decode
            operation: "encode" or "decode"
            format_type: base64, hex, url, etc.
        """
        try:
            import base64
            import urllib.parse
            import codecs
            
            # Remove whitespace for consistent processing
            input_data = input_data.strip()
            
            result = ""
            explanation = ""
            
            if operation == "encode":
                if format_type == "base64":
                    result = base64.b64encode(input_data.encode()).decode()
                    explanation = "Converted text to Base64 encoded string"
                elif format_type == "hex":
                    result = input_data.encode().hex()
                    explanation = "Converted text to hexadecimal representation"
                elif format_type == "url":
                    result = urllib.parse.quote(input_data)
                    explanation = "URL encoded the input string"
                elif format_type == "binary":
                    result = ' '.join(format(ord(c), '08b') for c in input_data)
                    explanation = "Converted text to binary representation"
                else:
                    return f"Unsupported encoding format: {format_type}"
            
            elif operation == "decode":
                try:
                    if format_type == "base64":
                        result = base64.b64decode(input_data).decode()
                        explanation = "Decoded Base64 string to text"
                    elif format_type == "hex":
                        # Handle both with and without spaces
                        clean_hex = input_data.replace(" ", "")
                        result = bytes.fromhex(clean_hex).decode()
                        explanation = "Converted hexadecimal to text"
                    elif format_type == "url":
                        result = urllib.parse.unquote(input_data)
                        explanation = "URL decoded the input string"
                    elif format_type == "binary":
                        # Handle binary with spaces
                        binary_values = input_data.split()
                        result = ''.join(chr(int(bin_val, 2)) for bin_val in binary_values)
                        explanation = "Converted binary to text"
                    else:
                        return f"Unsupported decoding format: {format_type}"
                except Exception as decode_error:
                    return f"Error decoding {format_type}: {str(decode_error)}. Please check that your input is valid {format_type} format."
            else:
                return f"Unsupported operation: {operation}. Use 'encode' or 'decode'."
                
            return {
                "result": result,
                "explanation": explanation,
                "operation": operation,
                "format": format_type
            }
            
        except Exception as e:
            return f"Error processing data: {str(e)}"
    
    def hash_analyzer(self, hash_value):
        """Analyze a hash value to determine type and provide information
        
        Args:
            hash_value: The hash string to analyze
        """
        try:
            hash_value = hash_value.strip()
            hash_length = len(hash_value)
            
            # Common hash types and their characteristics
            hash_types = {
                32: {
                    "name": "MD5",
                    "strength": "Weak - vulnerable to collision attacks",
                    "pattern": r"^[a-fA-F0-9]{32}$"
                },
                40: {
                    "name": "SHA-1",
                    "strength": "Weak - vulnerable to collision attacks",
                    "pattern": r"^[a-fA-F0-9]{40}$"
                },
                64: {
                    "name": "SHA-256",
                    "strength": "Strong - currently considered secure",
                    "pattern": r"^[a-fA-F0-9]{64}$"
                },
                96: {
                    "name": "SHA-384",
                    "strength": "Strong - currently considered secure",
                    "pattern": r"^[a-fA-F0-9]{96}$"
                },
                128: {
                    "name": "SHA-512",
                    "strength": "Very Strong - currently considered secure",
                    "pattern": r"^[a-fA-F0-9]{128}$"
                }
            }
            
            import re
            
            # Check if the hash matches the pattern for its length
            if hash_length in hash_types:
                hash_info = hash_types[hash_length]
                if re.match(hash_info["pattern"], hash_value):
                    result = {
                        "hash_type": hash_info["name"],
                        "length": hash_length,
                        "strength": hash_info["strength"],
                        "valid_format": True,
                        "additional_info": f"This appears to be a valid {hash_info['name']} hash."
                    }
                else:
                    result = {
                        "hash_type": "Unknown",
                        "length": hash_length,
                        "strength": "Unknown",
                        "valid_format": False,
                        "additional_info": "This string has the right length for some hash types but contains invalid characters."
                    }
            else:
                # Check for other hash types or formats
                if re.match(r"^\$2[ayb]\$[0-9]{2}\$[A-Za-z0-9./]{53}$", hash_value):
                    result = {
                        "hash_type": "bcrypt",
                        "length": hash_length,
                        "strength": "Strong - designed to be slow for password hashing",
                        "valid_format": True,
                        "additional_info": "This appears to be a bcrypt password hash."
                    }
                elif re.match(r"^\$6\$[a-zA-Z0-9./]{8,16}\$[a-zA-Z0-9./]{86}$", hash_value):
                    result = {
                        "hash_type": "SHA-512 crypt",
                        "length": hash_length,
                        "strength": "Strong",
                        "valid_format": True,
                        "additional_info": "This appears to be a SHA-512 crypt hash used in Linux/Unix systems."
                    }
                else:
                    result = {
                        "hash_type": "Unknown",
                        "length": hash_length,
                        "strength": "Unknown",
                        "valid_format": "Unknown",
                        "additional_info": "This string doesn't match common hash formats. It may be a custom hash format, encrypted data, or not a hash at all."
                    }
            
            return result
            
        except Exception as e:
            return f"Error analyzing hash: {str(e)}"
    
    def vulnerability_assessment(self, system_description):
        """Generate a vulnerability assessment based on a system description
        
        Args:
            system_description: Description of the system to assess
        """
        try:
            prompt = f"""Perform a vulnerability assessment for the following system:

            {system_description}
            
            In your assessment, include:
            1. Potential vulnerability categories based on the system description
            2. Specific vulnerabilities likely to affect this system
            3. OWASP Top 10 or SANS Top 25 vulnerabilities that apply
            4. Assessment methodology recommendations
            5. Testing approaches for each vulnerability category
            6. Mitigation strategies prioritized by risk
            7. Recommended security tools for testing
            
            Present this as a professional vulnerability assessment report.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating vulnerability assessment: {str(e)}"
    
    def generate_incident_response_plan(self, incident_type, organization_size):
        """Generate an incident response plan template
        
        Args:
            incident_type: Type of security incident (data breach, ransomware, etc.)
            organization_size: Size of the organization (small, medium, large)
        """
        try:
            prompt = f"""Generate a comprehensive incident response plan for a {organization_size} organization dealing with a {incident_type} incident.
            
            The plan should include:
            1. Incident response team structure and roles
            2. Preparation measures specific to {incident_type}
            3. Detection and identification procedures
            4. Containment strategies
            5. Eradication and recovery steps
            6. Post-incident analysis methodology
            7. Communication plan (internal and external)
            8. Legal and compliance considerations
            9. Documentation requirements
            10. Training recommendations
            
            Format this as a complete incident response playbook that could be implemented by a {organization_size} organization.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating incident response plan: {str(e)}"
    
    def analyze_code_security(self, code_snippet, language):
        """Analyze code for security vulnerabilities
        
        Args:
            code_snippet: The code to analyze
            language: Programming language of the code
        """
        try:
            prompt = f"""Perform a security code review on the following {language} code:
            
            ```{language}
            {code_snippet}
            ```
            
            In your analysis:
            1. Identify security vulnerabilities or weaknesses
            2. Categorize each issue by type (e.g., injection, authentication, etc.)
            3. Assess the risk level of each issue (Critical, High, Medium, Low)
            4. Provide specific line references where problems occur
            5. Explain why each issue is a security concern
            6. Recommend secure coding fixes with example code
            7. Suggest secure coding practices relevant to {language}
            
            Format your response as a professional security code review report.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing code security: {str(e)}"
    
    def generate_ctf_challenge(self, difficulty, category):
        """Generate a Capture The Flag (CTF) challenge for practice
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            category: Challenge category (web, crypto, forensics, etc.)
        """
        try:
            prompt = f"""Create a {difficulty} level {category} Capture The Flag (CTF) challenge for cybersecurity training.
            
            The challenge should include:
            1. A compelling scenario or background story
            2. Detailed challenge description
            3. Any necessary setup instructions or environment details
            4. Files or code that would need to be created (described in detail)
            5. Step-by-step solution guide (for the instructor)
            6. Hints that could be provided to participants (at least 3)
            7. Learning objectives and security concepts demonstrated
            8. Suggested point value for a CTF competition
            
            The challenge should be realistic, educational, and engaging while adhering to ethical standards.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating CTF challenge: {str(e)}"
    
    def explain_security_concept(self, concept_name):
        """Provide detailed explanation of security concepts with practical examples
        
        Args:
            concept_name: The security concept to explain
        """
        try:
            prompt = f"""Provide a comprehensive explanation of the security concept: {concept_name}
            
            In your explanation, include:
            1. Definition and core principles
            2. Historical context and development
            3. How this concept works technically
            4. Real-world applications and implementations
            5. Common attacks or vulnerabilities related to this concept
            6. Defensive strategies and best practices
            7. Practical examples that illustrate the concept
            8. Tools or frameworks associated with this concept
            9. Future trends or developments
            10. Resources for further learning
            
            Make the explanation accessible but technically accurate, suitable for cybersecurity training.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error explaining security concept: {str(e)}"
            
    def decrypt_or_encrypt_data(self, input_data, operation, algorithm, key=None, iv=None):
        """Encrypt or decrypt data using common cryptographic algorithms
        
        Args:
            input_data: Data to encrypt/decrypt
            operation: "encrypt" or "decrypt"
            algorithm: Algorithm to use (aes, des, etc.)
            key: Encryption/decryption key (will generate if none provided for encrypt)
            iv: Initialization vector (will generate if none provided for encrypt)
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            import os
            import base64
            
            supported_algorithms = {
                "aes": {
                    "key_sizes": [16, 24, 32],  # AES-128, AES-192, AES-256
                    "block_size": 16,
                    "algorithm": algorithms.AES
                },
                "des": {
                    "key_sizes": [8],
                    "block_size": 8,
                    "algorithm": algorithms.TripleDES
                },
                "chacha20": {
                    "key_sizes": [32],
                    "nonce_size": 16,
                    "algorithm": algorithms.ChaCha20
                }
            }
            
            if algorithm.lower() not in supported_algorithms:
                return f"Unsupported algorithm: {algorithm}. Supported algorithms are: {', '.join(supported_algorithms.keys())}"
            
            algo_info = supported_algorithms[algorithm.lower()]
            
            # Handling key
            if operation == "encrypt":
                if key is None:
                    # Generate a random key of the first valid size
                    key = os.urandom(algo_info["key_sizes"][0])
                    key_b64 = base64.b64encode(key).decode()
                else:
                    # Use provided key, ensuring correct size
                    try:
                        key_bytes = base64.b64decode(key)
                        if len(key_bytes) not in algo_info["key_sizes"]:
                            return f"Invalid key size for {algorithm}. Valid sizes (bytes): {algo_info['key_sizes']}"
                        key = key_bytes
                        key_b64 = key
                    except:
                        return "Invalid key format. Key must be base64 encoded."
            else:  # Decrypt
                if key is None:
                    return "Decryption requires a key."
                try:
                    key = base64.b64decode(key)
                    if len(key) not in algo_info["key_sizes"]:
                        return f"Invalid key size for {algorithm}. Valid sizes (bytes): {algo_info['key_sizes']}"
                except:
                    return "Invalid key format. Key must be base64 encoded."
            
            # AES and DES require a mode (CBC, GCM, etc.)
            if algorithm.lower() in ["aes", "des"]:
                # Handle IV for block ciphers
                if operation == "encrypt":
                    if iv is None:
                        iv = os.urandom(algo_info["block_size"])
                        iv_b64 = base64.b64encode(iv).decode()
                    else:
                        try:
                            iv_bytes = base64.b64decode(iv)
                            if len(iv_bytes) != algo_info["block_size"]:
                                return f"Invalid IV size for {algorithm}. Required size: {algo_info['block_size']} bytes"
                            iv = iv_bytes
                            iv_b64 = iv
                        except:
                            return "Invalid IV format. IV must be base64 encoded."
                else:  # Decrypt
                    if iv is None:
                        return f"{algorithm} decryption in CBC mode requires an IV."
                    try:
                        iv = base64.b64decode(iv)
                        if len(iv) != algo_info["block_size"]:
                            return f"Invalid IV size for {algorithm}. Required size: {algo_info['block_size']} bytes"
                    except:
                        return "Invalid IV format. IV must be base64 encoded."
                
                # Create cipher
                cipher = Cipher(
                    algo_info["algorithm"](key),
                    modes.CBC(iv),
                    backend=default_backend()
                )
            
            elif algorithm.lower() == "chacha20":
                if operation == "encrypt":
                    if iv is None:
                        iv = os.urandom(algo_info["nonce_size"])
                        iv_b64 = base64.b64encode(iv).decode()
                    else:
                        try:
                            iv_bytes = base64.b64decode(iv)
                            if len(iv_bytes) != algo_info["nonce_size"]:
                                return f"Invalid nonce size for ChaCha20. Required size: {algo_info['nonce_size']} bytes"
                            iv = iv_bytes
                            iv_b64 = iv
                        except:
                            return "Invalid nonce format. Nonce must be base64 encoded."
                else:  # Decrypt
                    if iv is None:
                        return "ChaCha20 decryption requires a nonce."
                    try:
                        iv = base64.b64decode(iv)
                        if len(iv) != algo_info["nonce_size"]:
                            return f"Invalid nonce size for ChaCha20. Required size: {algo_info['nonce_size']} bytes"
                    except:
                        return "Invalid nonce format. Nonce must be base64 encoded."
                
                # Create cipher
                cipher = Cipher(
                    algo_info["algorithm"](key, iv),
                    mode=None,  # ChaCha20 doesn't use a mode
                    backend=default_backend()
                )
            
            # Perform encryption/decryption
            if operation == "encrypt":
                # Add padding for block ciphers
                if algorithm.lower() in ["aes", "des"]:
                    from cryptography.hazmat.primitives import padding
                    padder = padding.PKCS7(algo_info["block_size"] * 8).padder()
                    data = padder.update(input_data.encode()) + padder.finalize()
                else:
                    data = input_data.encode()
                
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data) + encryptor.finalize()
                result = base64.b64encode(ciphertext).decode()
                
                return {
                    "result": result,
                    "key": key_b64 if 'key_b64' in locals() else base64.b64encode(key).decode(),
                    "iv": iv_b64 if 'iv_b64' in locals() else base64.b64encode(iv).decode() if iv else None,
                    "algorithm": algorithm,
                    "operation": "encrypt"
                }
                
            else:  # Decrypt
                try:
                    ciphertext = base64.b64decode(input_data)
                    decryptor = cipher.decryptor()
                    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
                    
                    # Remove padding for block ciphers
                    if algorithm.lower() in ["aes", "des"]:
                        from cryptography.hazmat.primitives import padding
                        unpadder = padding.PKCS7(algo_info["block_size"] * 8).unpadder()
                        decrypted_data = unpadder.update(decrypted_data) + unpadder.finalize()
                    
                    return {
                        "result": decrypted_data.decode(),
                        "algorithm": algorithm,
                        "operation": "decrypt"
                    }
                except Exception as decrypt_error:
                    return f"Decryption error: {str(decrypt_error)}. Check that your key, IV, and ciphertext are correct."
            
        except Exception as e:
            return f"Error in cryptographic operation: {str(e)}"
    
    
def main():
    
    st.set_page_config(
        page_title="CyberGeni AI", 
        page_icon="🔐", 
        layout="wide"
    )

    st.markdown("""
        <style>
        /* Base styles and typography */
        body {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }

        /* Main header styling */
        .main-header {
            padding: 1.5rem;
            background-color: #1e293b;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Status indicator */
        .status-indicator {
            width: 12px;
            height: 12px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            box-shadow: 0 0 0 rgba(16, 185, 129, 0.4);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        .glow-text {
            color: #10b981;
            font-weight: 500;
            text-shadow: 0 0 5px rgba(16, 185, 129, 0.5);
        }

        /* Card styling */
        .sleek-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Category pills */
        .category-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background-color: #1e293b;
            color: #94a3b8;
            border-radius: 9999px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #334155;
            transition: all 0.3s ease;
        }

        .category-pill:hover {
            background-color: #334155;
            color: #e2e8f0;
            cursor: pointer;
            border-color: #475569;
        }

        /* Chat container */
        .chat-container {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            height: 500px;
            overflow-y: auto;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Message styling */
        .message-container {
            display: flex;
            margin-bottom: 1.5rem;
            gap: 12px;
        }

        .user-container {
            justify-content: flex-end;
        }

        .assistant-container {
            justify-content: flex-start;
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.875rem;
            flex-shrink: 0;
        }

        .user-avatar {
            background-color: #7c3aed;
            color: white;
        }

        .assistant-avatar {
            background-color: #10b981;
            color: white;
        }

        .message-box {
            padding: 1rem;
            border-radius: 12px;
            max-width: 90%;
            min-width: 100px;
        }

        .user-message {
            background-color: #7c3aed;
            color: white;
            border-top-right-radius: 0;
        }

        .assistant-message {
            background-color: #334155;
            color: #e2e8f0;
            border-top-left-radius: 0;
        }

        /* Typing indicator animation */
        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 1rem;
            background-color: #334155;
            border-radius: 12px;
            border-top-left-radius: 0;
            min-width: 60px;
        }

        .typing-dot {
            display: block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #94a3b8;
            margin-right: 5px;
            animation: bounce 1.3s linear infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.15s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.3s;
            margin-right: 0;
        }

        @keyframes bounce {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-5px);
            }
        }

        /* Button and input customization */
        .stButton>button {
            background-color: #10b981;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #059669;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: #1e293b;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background-color: #475569;
            border-radius: 20px;
            border: 2px solid #1e293b;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }

        /* Radio button styling for tabs */
        div.row-widget.stRadio > div {
            flex-direction: row;
            gap: 10px;
        }

        div.row-widget.stRadio > div > label {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 6px 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-weight: 500;
            border: 1px solid #334155;
            transition: all 0.2s ease;
        }

        div.row-widget.stRadio > div > label:hover {
            background-color: #334155;
            color: #e2e8f0;
            cursor: pointer;
        }

        div.row-widget.stRadio > div[role="radiogroup"] input[type="radio"]:checked + label {
            background-color: #10b981;
            color: white;
            border-color: #059669;
        }

        /* Code block styling */
        code {
            background-color: #0f172a;
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
        }

        pre {
            background-color: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            border: 1px solid #334155;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    # Add custom CSS for modern UI - Kept original CSS from your code

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False
        
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
        
    # Header with cyber security themed logo
    st.markdown("""
        <div class="main-header">
            <h1 style="font-size: 2.75rem; color: #e2e8f0; margin-bottom: 0.5rem;">CyberGeni AI</h1>
            <p style="color: #a0aec0; font-size: 1.1rem;">
                AI-powered cybersecurity training assistant for ethical hacking education by gamkers
            </p>
            <div style="margin-top: 1rem; display: flex; align-items: center;">
                <div class="status-indicator"></div>
                <span class="glow-text">Active & Secure</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <h3 style="display: flex; align-items: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
                Cybersecurity Training
            </h3>
        """, unsafe_allow_html=True)
        
        # Initialize button with animation
        if st.button("Initialize Assistant", key="init_assistant"):
            try:
                # Get API key from secrets.toml or environment variable
                groq_api_key = "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"
                
                # Initialize the assistant
                st.session_state.assistant = CyberSecurityAssistant(groq_api_key=groq_api_key)
                st.success("CyberGeni initialized successfully!")
                
            except Exception as e:
                st.error(f"Error initializing assistant: {e}")
        
        # Search tips and categories
        st.markdown("""
            <h3 style="display: flex; align-items: center; margin-top: 2rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                </svg>
                Features & Categories
            </h3>
            
            <div style="margin-top: 1rem;">
                <div class="category-pill">Network Security</div>
                <div class="category-pill">Web App Security</div>
                <div class="category-pill">Encryption</div>
                <div class="category-pill">Malware Analysis</div>
                <div class="category-pill">Penetration Testing</div>
                <div class="category-pill">Social Engineering</div>
                <div class="category-pill">Cloud Security</div>
                <div class="category-pill">IoT Security</div>
            </div>
            
            <h3 style="display: flex; align-items: center; margin-top: 2rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                New Features
            </h3>
            
            <div style="background-color: #1a1e24; padding: 1.25rem; border-radius: 12px; margin-top: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <p style="font-weight: 600; margin-bottom: 0.75rem; color: #e2e8f0;">Try these new features:</p>
                <ul style="margin-left: 0.75rem; padding-left: 1rem; color: #a0aec0;">
                    <li style="margin-bottom: 0.5rem;">CVE Database Search</li>
                    <li style="margin-bottom: 0.5rem;">Latest Hacker News</li>
                    <li style="margin-bottom: 0.5rem;">Security Tool Commands</li>
                    <li style="margin-bottom: 0.5rem;">Google Dorking Generator</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Clear conversation button
        if st.button("Clear Conversation", key="clear_conv"):
            st.session_state.messages = []
            if st.session_state.assistant:
                st.session_state.assistant.conversation_history = []
            st.rerun()
            
        # Disclaimer
        st.markdown("""
            <div style="background-color: #7f1d1d; padding: 1rem; border-radius: 8px; margin-top: 2rem;">
                <h4 style="color: #fee2e2; margin-top: 0;">⚠️ Educational Use Only</h4>
                <p style="color: #fecaca; font-size: 0.85rem; margin-bottom: 0;">
                    All information provided by CyberGeni is for educational purposes only. Always practice ethical hacking and only test systems you own or have explicit permission to test.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content - Tabs system
    tabs = ["Chat", "CVE Database", "Security News", "Tool Commands", "Google Dorking"]
    st.session_state.current_tab = st.radio("Select Feature", tabs, horizontal=True)
    
    # Chat Tab
    if st.session_state.current_tab == "Chat":
        # Feature cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="sleek-card">
                    <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                            <circle cx="12" cy="7" r="4"></circle>
                        </svg>
                        Ethical Hacking Training
                    </h3>
                    <p style="color: #a0aec0; margin-bottom: 0;">Learn ethical hacking principles, tools, and methodologies through guided training. Get personalized instructions and hands-on practice for penetration testing and vulnerability assessment.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="sleek-card">
                    <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                        </svg>
                        Security Best Practices
                    </h3>
                    <p style="color: #a0aec0; margin-bottom: 0;">Learn defensive security measures, incident response protocols, and security best practices to protect systems and networks from cyber threats.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Chat container
        # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Function to display messages
        def display_messages():
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="message-container user-container">
                            <div class="avatar user-avatar">U</div>
                            <div class="message-box user-message">{message['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif message["role"] == "assistant":
                    st.markdown(
                        f"""
                        <div class="message-container assistant-container">
                            <div class="avatar assistant-avatar">C</div>
                            <div class="message-box assistant-message">{message['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            # Display thinking animation if assistant is processing
            if st.session_state.thinking:
                st.markdown(
                    """
                    <div class="message-container assistant-container">
                        <div class="avatar assistant-avatar">C</div>
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
        # Display previous messages
        display_messages()
        
        # Initial hint if no messages
        if not st.session_state.messages:
            st.markdown("""
                <div style="text-align: center; padding: 32px 16px; color: #6b7280;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto 16px;">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    <p>No messages yet. Initialize the assistant and start the conversation!</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User input and processing
        if prompt := st.chat_input("Ask CyberGeni...", key="user_input"):
            if st.session_state.assistant is None:
                st.warning("Please initialize the assistant first using the button in the sidebar.")
            else:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Set thinking state to true and rerun to show the animation
                st.session_state.thinking = True
                st.rerun()
        
        # This part only runs after the rerun if thinking is True
        if st.session_state.thinking and st.session_state.assistant is not None:
            try:
                # Get assistant response
                response = st.session_state.assistant.query(st.session_state.messages[-1]["content"])
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Stop thinking, rerun to show updated chat
                st.session_state.thinking = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.session_state.thinking = False
                st.rerun()
    
    # CVE Database Tab
    elif st.session_state.current_tab == "CVE Database":
        st.markdown("""
            <div class="sleek-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    CVE Database Search
                </h3>
                <p style="color: #a0aec0; margin-bottom: 1rem;">Search for Common Vulnerabilities and Exposures (CVE) by ID or keyword. Get detailed information about security vulnerabilities, including severity, impact, and remediation.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # CVE search form
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            cve_query = st.text_input("Enter CVE ID or keyword", 
                                      placeholder="Example: CVE-2021-44228 or log4j")
            
        with search_col2:
            cve_limit = st.selectbox("Result limit", [5, 10, 15, 20], index=0)
            search_button = st.button("Search CVE", key="search_cve_button")
        
        # Search process
        if search_button and cve_query:
            if st.session_state.assistant is None:
                st.warning("Please initialize the assistant first using the button in the sidebar.")
            else:
                with st.spinner("Searching CVE database..."):
                    cve_results = st.session_state.assistant.search_cve(cve_query, cve_limit)
                    
                    if isinstance(cve_results, str):
                        st.warning(cve_results)
                    else:
                        # Display results in a nice format
                        for result in cve_results:
                            with st.expander(f"{result['cve_id']} - Severity: {result['severity']} ({result['base_score']})"):
                                st.markdown(f"### {result['cve_id']}")
                                
                                # Create two columns for severity and dates
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Severity**: {result['severity']}")
                                    st.markdown(f"**CVSS Score**: {result['base_score']}")
                                
                                with col2:
                                    st.markdown(f"**Published**: {result['published']}")
                                    st.markdown(f"**Last Modified**: {result['last_modified']}")
                                
                                st.markdown("### Description")
                                st.markdown(result['description'])
                                
                                # References
                                if result['references']:
                                    st.markdown("### References")
                                    for ref in result['references'][:5]:  # Limit to first 5 references
                                        st.markdown(f"- [{ref.get('url', 'Link')}]({ref.get('url', '#')})")
    
    # Security News Tab
    elif st.session_state.current_tab == "Security News":
        st.markdown("""
            <div class="sleek-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                        <path d="M19 20H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h10l6 6v8a2 2 0 0 1-2 2z"></path>
                        <line x1="12" y1="18" x2="12" y2="12"></line>
                        <line x1="9" y1="15" x2="15" y2="15"></line>
                    </svg>
                    Latest Cybersecurity News
                </h3>
                <p style="color: #a0aec0; margin-bottom: 1rem;">Stay updated with the latest cybersecurity news from The Hacker News, summarized by AI for quick insights into current threats, vulnerabilities, and industry developments.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # News fetch options
        news_col1, news_col2 = st.columns([3, 1])
        
        with news_col2:
            news_limit = st.selectbox("Article limit", [3, 5, 7, 10], index=1)
            news_button = st.button("Fetch News", key="fetch_news_button")
        
        # News fetch process
        if news_button:
            if st.session_state.assistant is None:
                st.warning("Please initialize the assistant first using the button in the sidebar.")
            else:
                with st.spinner("Fetching and summarizing latest cybersecurity news..."):
                    news_results = st.session_state.assistant.get_hacker_news(news_limit)
                    
                    if isinstance(news_results, str):
                        st.warning(news_results)
                    else:
                        # Display news in a nice format
                        for i, news in enumerate(news_results):
                            with st.container():
                                st.markdown(f"""
                                <div class="sleek-card" style="margin-bottom: 1rem;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: #e2e8f0;">{news['title']}</h4>
                                    <p style="color: #a0aec0; font-size: 0.8rem; margin-bottom: 0.75rem;">Published: {news['published']}</p>
                                    <p style="color: #e2e8f0; margin-bottom: 1rem;">{news['summary']}</p>
                                    <a href="{news['link']}" target="_blank" style="color: #10b981; text-decoration: none; font-weight: 500;">Read full article →</a>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Tool Commands Tab
    elif st.session_state.current_tab == "Tool Commands":
        st.markdown("""
            <div class="sleek-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                    </svg>
                    Security Tool Commands
                </h3>
                <p style="color: #a0aec0; margin-bottom: 1rem;">Generate practical command examples for common security tools. Learn how to use tools like Nmap, Metasploit, Wireshark, and more for your security assessments.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Tool commands form
        tool_col1, tool_col2 = st.columns(2)
        
        with tool_col1:
            tool_name = st.text_input("Enter security tool name", placeholder="Example: nmap, metasploit, burpsuite")
            
        with tool_col2:
            task_desc = st.text_input("What do you want to achieve with this tool?", 
                                      placeholder="Example: network scanning, discovery of web vulnerabilities")
        
        tool_button = st.button("Generate Commands", key="gen_tool_button")
        
        # Tool command generation process
        if tool_button and tool_name and task_desc:
            if st.session_state.assistant is None:
                st.warning("Please initialize the assistant first using the button in the sidebar.")
            else:
                with st.spinner(f"Generating {tool_name} commands for {task_desc}..."):
                    tool_results = st.session_state.assistant.generate_tool_commands(tool_name, task_desc)
                    # st.write(tool_results)
                    st.markdown(f"""
                        <div class="sleek-card" style="background-color: #0f172a; border: 1px solid #1e293b;">
                            <h4 style="margin: 0 0 1rem 0; color: #e2e8f0; display: flex; align-items: center;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                                    <polyline points="9 11 12 14 22 4"></polyline>
                                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                                </svg>
                                Generated Commands: {tool_name}
                            </h4>
                            <div style="color: #e2e8f0; font-family: monospace; background-color: #1e293b; padding: 1rem; border-radius: 8px; white-space: pre-wrap;">
                                {tool_results}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    # Google Dorking Tab
    elif st.session_state.current_tab == "Google Dorking":
        st.markdown("""
            <div class="sleek-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #e2e8f0; display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    Google Dorking Generator
                </h3>
                <p style="color: #a0aec0; margin-bottom: 1rem;">Generate Google dorking queries for finding specific information on websites. Learn advanced search operators to identify potential security vulnerabilities and improve targeted reconnaissance.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Google dork form
        dork_col1, dork_col2 = st.columns(2)
        
        with dork_col1:
            target_type = st.text_input("What are you looking for?", 
                                       placeholder="Example: exposed databases, login pages, config files")
            
        with dork_col2:
            dork_objective = st.text_input("What's your security objective?", 
                                         placeholder="Example: vulnerability assessment, information gathering")
        
        dork_button = st.button("Generate Google Dorks", key="gen_dork_button")
        
        # Dork generation process
        if dork_button and target_type and dork_objective:
            if st.session_state.assistant is None:
                st.warning("Please initialize the assistant first using the button in the sidebar.")
            else:
                with st.spinner(f"Generating Google dorks for finding {target_type}..."):
                    dork_results = st.session_state.assistant.generate_google_dorks(target_type, dork_objective)
                    
                    st.write(dork_results)

                    # st.markdown(f"""
                    #     <div class="sleek-card" style="background-color: #0f172a; border: 1px solid #1e293b;">
                    #         <h4 style="margin: 0 0 1rem 0; color: #e2e8f0; display: flex; align-items: center;">
                    #             <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                    #                 <polyline points="9 11 12 14 22 4"></polyline>
                    #                 <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                    #             </svg>
                    #             Generated Google Dorks
                    #         </h4>
                    #         <div style="color: #e2e8f0; font-family: monospace; background-color: #1e293b; padding: 1rem; border-radius: 8px; white-space: pre-wrap;">
                    #             {dork_results}
                    #         </div>
                    #         <div style="background-color: #851c1c; padding: 0.75rem; border-radius: 8px; margin-top: 1rem;">
                    #             <p style="color: #fee2e2; font-size: 0.8rem; margin: 0;">
                    #                 <strong>⚠️ Ethical Reminder:</strong> Only use these dorks on websites you own or have explicit permission to test. Using these techniques without permission may violate laws and terms of service.
                    #             </p>
                    #         </div>
                    #     </div>
                    # """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
