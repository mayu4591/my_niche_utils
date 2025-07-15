import { api } from "../../scripts/api.js";

// MyNicheUtilsメッセージング機能
class MyNicheUtilsMessaging {
    static sendMessage(id, message) {
        const body = new FormData();
        body.append('message', message);
        body.append('id', id);

        return api.fetchApi("/my_niche_utils_message", {
            method: "POST",
            body
        }).then(response => {
            console.log(`MyNicheUtils: Message sent successfully - ID: ${id}, Message: ${message}`);
            return response;
        }).catch(error => {
            console.error(`MyNicheUtils: Failed to send message - ID: ${id}, Message: ${message}`, error);
            throw error;
        });
    }

    static sendContinue(nodeId) {
        return this.sendMessage(nodeId, 'continue');
    }

    static sendCancel(nodeId) {
        return this.sendMessage(nodeId, 'cancel');
    }
}

export { MyNicheUtilsMessaging };
